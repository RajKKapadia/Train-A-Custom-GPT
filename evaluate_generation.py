import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken

from src.config import CONFIG
from src.create_model import GPT


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

    generated_new_tokens = 0
    ended_with_eos = False

    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -CONFIG.model.block_size :]

        with torch.no_grad():
            logits, _ = model(input_cond)

        logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = top_k_filter(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)
        generated_new_tokens += 1

        if eos_token_id is not None and next_id.item() == eos_token_id:
            ended_with_eos = True
            break

    return input_ids, generated_new_tokens, ended_with_eos


def load_model(checkpoint_path: str | Path, device: str):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = GPT(CONFIG.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, checkpoint


def get_ngrams(tokens: list[str], n: int):
    if len(tokens) < n:
        return []

    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(tokens: list[str], n: int) -> float:
    ngrams = get_ngrams(tokens, n)

    if not ngrams:
        return 0.0

    return len(set(ngrams)) / len(ngrams)


def unique_ngram_ratio(tokens: list[str], n: int) -> float:
    ngrams = get_ngrams(tokens, n)

    if not ngrams:
        return 0.0

    return len(set(ngrams)) / len(ngrams)


def repeated_ngram_count(tokens: list[str], n: int) -> int:
    ngrams = get_ngrams(tokens, n)

    if not ngrams:
        return 0

    counts = Counter(ngrams)
    return sum(count - 1 for count in counts.values() if count > 1)


def simple_word_tokens(text: str) -> list[str]:
    return text.lower().replace("\n", " ").split()


def evaluate_text(generated_text: str):
    words = simple_word_tokens(generated_text)

    return {
        "word_count": len(words),
        "distinct_1": distinct_n(words, 1),
        "distinct_2": distinct_n(words, 2),
        "unique_4gram_ratio": unique_ngram_ratio(words, 4),
        "repeated_4gram_count": repeated_ngram_count(words, 4),
    }


def read_prompts(path: str | Path) -> list[str]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    prompts = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)

    if not prompts:
        raise ValueError(f"No prompts found in: {path}")

    return prompts


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_samples_text(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write("=" * 100 + "\n")
            f.write(f"Prompt ID: {row['prompt_id']}\n")
            f.write(f"Prompt: {row['prompt']}\n")
            f.write(f"Generated tokens: {row['generated_tokens']}\n")
            f.write(f"Ended with EOS: {row['ended_with_eos']}\n")
            f.write("-" * 100 + "\n")
            f.write(row["generated_text"].strip())
            f.write("\n\n")


def summarize(rows: list[dict]) -> dict:
    total = len(rows)

    if total == 0:
        return {}

    def avg(key: str):
        return sum(float(row[key]) for row in rows) / total

    eos_count = sum(1 for row in rows if row["ended_with_eos"])

    return {
        "num_prompts": total,
        "eos_success_rate": eos_count / total,
        "avg_generated_tokens": avg("generated_tokens"),
        "avg_word_count": avg("word_count"),
        "avg_distinct_1": avg("distinct_1"),
        "avg_distinct_2": avg("distinct_2"),
        "avg_unique_4gram_ratio": avg("unique_4gram_ratio"),
        "avg_repeated_4gram_count": avg("repeated_4gram_count"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated samples from TinyStories GPT model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/tinystories-gpt/best.pt",
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default="eval_prompts.txt",
        help="Path to eval prompts file.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to checkpoint folder / generation_eval.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Maximum new tokens per prompt.",
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
        "--num-samples-per-prompt",
        type=int,
        default=1,
        help="Generate N samples for each prompt.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=CONFIG.train.device,
        choices=["cuda", "cpu"],
        help="Device to run generation on.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    checkpoint_path = Path(args.checkpoint)

    if args.out_dir is None:
        out_dir = checkpoint_path.parent / "generation_eval"
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(CONFIG.dataset.tokenizer_name)

    model, checkpoint = load_model(checkpoint_path, device)
    prompts = read_prompts(args.prompts)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint iter: {checkpoint.get('iter_num')}")
    print(f"Best valid loss: {checkpoint.get('best_valid_loss')}")
    print(f"Prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.num_samples_per_prompt}")
    print(f"Output dir: {out_dir}")
    print("-" * 100)

    rows = []

    sample_id = 0

    for prompt_id, prompt in enumerate(prompts):
        for sample_index in range(args.num_samples_per_prompt):
            sample_id += 1

            prompt_ids = enc.encode(prompt)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            output_ids, generated_tokens, ended_with_eos = generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=enc.eot_token,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            full_text = enc.decode(output_ids[0].tolist())
            full_text_clean = full_text.replace("<|endoftext|>", "")

            # Only the newly generated part.
            generated_ids = output_ids[0].tolist()[len(prompt_ids) :]
            generated_text = enc.decode(generated_ids)
            generated_text_clean = generated_text.replace("<|endoftext|>", "")

            text_metrics = evaluate_text(generated_text_clean)

            row = {
                "sample_id": sample_id,
                "prompt_id": prompt_id,
                "sample_index": sample_index,
                "prompt": prompt,
                "generated_text": generated_text_clean.strip(),
                "full_text": full_text_clean.strip(),
                "generated_tokens": generated_tokens,
                "ended_with_eos": ended_with_eos,
                "temperature": args.temperature,
                "top_k": args.top_k,
                **text_metrics,
            }

            rows.append(row)

            print(
                f"[{sample_id}] prompt_id={prompt_id}, "
                f"tokens={generated_tokens}, "
                f"eos={ended_with_eos}, "
                f"distinct_2={row['distinct_2']:.3f}, "
                f"unique_4gram={row['unique_4gram_ratio']:.3f}"
            )

    summary = summarize(rows)

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iter": checkpoint.get("iter_num"),
        "best_valid_loss": checkpoint.get("best_valid_loss"),
        "config": CONFIG.to_dict(),
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "num_samples_per_prompt": args.num_samples_per_prompt,
            "seed": args.seed,
        },
        "summary": summary,
        "samples": rows,
    }

    save_json(out_dir / "generation_eval.json", result)
    save_csv(out_dir / "generation_eval.csv", rows)
    save_samples_text(out_dir / "generation_samples.txt", rows)

    print("-" * 100)
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print("-" * 100)
    print(f"Saved JSON: {out_dir / 'generation_eval.json'}")
    print(f"Saved CSV: {out_dir / 'generation_eval.csv'}")
    print(f"Saved samples: {out_dir / 'generation_samples.txt'}")


if __name__ == "__main__":
    main()
