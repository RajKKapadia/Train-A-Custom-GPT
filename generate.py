import argparse
from pathlib import Path

import torch
import tiktoken

from src.config import CONFIG
from src.create_model import GPT, model_config_from_dict


def generate(
    model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 0.8,
    top_k: int | None = 50,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 4,
    use_cache: bool = True,
):
    model.eval()
    return model.generate(
        idx=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        use_cache=use_cache,
    )


def load_model(checkpoint_path: str | Path, device: str):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_cfg = model_config_from_dict(
        checkpoint.get("model_config", CONFIG.model.__dict__)
    )
    model = GPT(model_cfg)

    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, checkpoint


def resolve_checkpoint(checkpoint_path: str | Path | None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    configured_path = Path(CONFIG.test.checkpoint_path)
    if configured_path.exists():
        return configured_path

    run_root = Path(CONFIG.train.out_dir)
    candidates = sorted(
        run_root.glob("*/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    return configured_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from trained TinyStories GPT model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. Defaults to the latest experiment best.pt.",
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
        default=200,
        help="Number of new tokens to generate.",
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
        help="Penalty for tokens already present in the prompt/context.",
    )

    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=CONFIG.test.no_repeat_ngram_size,
        help="Block any token that would repeat an n-gram of this size. Use 0 to disable.",
    )

    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV-cache generation and recompute the full context every token.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=CONFIG.train.device,
        choices=["cuda", "cpu"],
        help="Device to run generation on.",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device = args.device

    enc = tiktoken.get_encoding(CONFIG.dataset.tokenizer_name)

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model, checkpoint = load_model(checkpoint_path, device)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint iter: {checkpoint.get('iter_num')}")
    print(f"Best valid loss: {checkpoint.get('best_valid_loss')}")
    print("-" * 80)

    prompt_ids = enc.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=enc.eot_token,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        use_cache=not args.no_kv_cache,
    )

    generated_text = enc.decode(output_ids[0].tolist())

    generated_text = generated_text.replace("<|endoftext|>", "")

    print(generated_text)


if __name__ == "__main__":
    main()
