import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken

from src.config import CONFIG
from src.create_model import GPT, model_config_from_dict


def top_k_filter(logits: torch.Tensor, top_k: int | None):
    if top_k is None or top_k <= 0:
        return logits

    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    min_value = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_value, torch.full_like(logits, float("-inf")), logits
    )


def generate(
    model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 0.8,
    top_k: int | None = 50,
):
    model.eval()

    for _ in range(max_new_tokens):
        # Crop context if it becomes longer than block size
        input_cond = input_ids[:, -model.config.block_size :]

        with torch.no_grad():
            logits, _ = model(input_cond)

        # Take logits from the last position
        logits = logits[:, -1, :]

        if temperature <= 0:
            # Greedy decoding
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = top_k_filter(logits, top_k)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        # Stop if EOS token is generated
        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    return input_ids


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
        default=0.8,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling. Use 0 to disable.",
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
    )

    generated_text = enc.decode(output_ids[0].tolist())

    generated_text = generated_text.replace("<|endoftext|>", "")

    print(generated_text)


if __name__ == "__main__":
    main()
