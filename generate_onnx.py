"""Generate text with an exported TinyStories GPT ONNX model."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import tiktoken

from src.config import CONFIG
from src.create_model import sample_next_token


def metadata_path_for(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(onnx_path.suffix + ".json")


def load_metadata(onnx_path: Path) -> dict[str, Any]:
    metadata_path = metadata_path_for(onnx_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"ONNX metadata not found: {metadata_path}. "
            "Export with export_onnx.py first."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def make_session(onnx_path: Path, provider: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if provider not in available:
        raise ValueError(
            f"Provider {provider!r} is not available. Available providers: {available}"
        )
    return ort.InferenceSession(str(onnx_path), providers=[provider])


def generate(
    session: ort.InferenceSession,
    input_ids: torch.Tensor,
    block_size: int,
    max_new_tokens: int,
    eos_token_id: int | None,
    temperature: float,
    top_k: int | None,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        input_cond = generated[:, -block_size:]
        (logits_np,) = session.run(
            ["logits"],
            {"input_ids": input_cond.cpu().numpy().astype(np.int64)},
        )
        logits = torch.from_numpy(logits_np)
        idx_next = sample_next_token(
            logits=logits,
            input_ids=generated.cpu(),
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        generated = torch.cat((generated, idx_next.to(generated.device)), dim=1)

        if eos_token_id is not None and bool(torch.all(idx_next == eos_token_id)):
            break

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text from an exported TinyStories GPT ONNX model."
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to model.onnx exported by export_onnx.py.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Prompt text to start generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=CONFIG.test.max_new_tokens,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=CONFIG.test.temperature,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=CONFIG.test.top_k,
        help="Top-k sampling. Use 0 to disable.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=CONFIG.test.repetition_penalty,
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=CONFIG.test.no_repeat_ngram_size,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.test.seed,
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="ONNX Runtime provider to use.",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    torch.manual_seed(args.seed)
    metadata = load_metadata(onnx_path)
    model_config = metadata["model_config"]
    tokenizer_name = metadata.get("tokenizer_name", CONFIG.dataset.tokenizer_name)
    block_size = int(model_config["block_size"])

    session = make_session(onnx_path, args.provider)
    enc = tiktoken.get_encoding(tokenizer_name)

    prompt_ids = enc.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    output_ids = generate(
        session=session,
        input_ids=input_ids,
        block_size=block_size,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=enc.eot_token,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    generated_text = enc.decode(output_ids[0].tolist()).replace("<|endoftext|>", "")

    print(f"Loaded ONNX model: {onnx_path}")
    print(f"Checkpoint: {metadata.get('checkpoint')}")
    print(f"Provider: {args.provider}")
    print("-" * 80)
    print(generated_text)


if __name__ == "__main__":
    main()
