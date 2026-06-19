"""Evaluate test loss and generate samples from the trained model."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import json
import math

import numpy as np
import torch
import tiktoken

from src.config import CONFIG, AppConfig
from src.create_model import create_model, model_config_from_dict
from src.train import BinDataset


@torch.no_grad()
def evaluate_test_loss(model, test_data, cfg: AppConfig, ctx):
    model.eval()
    losses = torch.zeros(cfg.test.num_eval_batches)
    for k in range(cfg.test.num_eval_batches):
        X, Y = test_data.get_batch(cfg.train.batch_size)
        with ctx:
            _, loss = model(X, Y)
        losses[k] = loss.item()
    mean_loss = losses.mean().item()
    return {
        "test_loss": mean_loss,
        "test_perplexity": math.exp(mean_loss) if mean_loss < 20 else float("inf"),
    }


@torch.no_grad()
def generate_samples(model, cfg: AppConfig, prompts: list[str]):
    enc = tiktoken.get_encoding(cfg.dataset.tokenizer_name)
    model.eval()
    results = []

    for prompt in prompts:
        ids = enc.encode_ordinary(prompt)
        x = torch.tensor(ids, dtype=torch.long, device=cfg.train.device)[None, ...]
        y = model.generate(
            x,
            max_new_tokens=cfg.test.max_new_tokens,
            temperature=cfg.test.temperature,
            top_k=cfg.test.top_k,
        )
        generated = enc.decode(y[0].tolist())
        completion = enc.decode(y[0][x.shape[1] :].tolist())
        results.append(
            {
                "prompt": prompt,
                "completion": completion,
                "full_text": generated,
            }
        )
    return results


def load_prompts_from_test_bin(
    cfg: AppConfig,
    dataset_dir: str | Path | None = None,
    block_size: int | None = None,
):
    enc = tiktoken.get_encoding(cfg.dataset.tokenizer_name)
    processed_dir = (
        Path(dataset_dir)
        if dataset_dir is not None
        else cfg.dataset.resolved_processed_dir()
    )
    data = np.memmap(processed_dir / "test.bin", dtype=np.uint16, mode="r")
    prompts = []
    step = max(1, len(data) // max(1, cfg.test.num_generation_prompts))
    prompt_len = min(40, (block_size or cfg.model.block_size) // 4)
    for i in range(0, len(data) - prompt_len, step):
        if len(prompts) >= cfg.test.num_generation_prompts:
            break
        prompts.append(enc.decode(data[i : i + prompt_len].astype(np.int64).tolist()))
    return prompts


def test(
    cfg: AppConfig = CONFIG,
    dataset_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    results_path: str | Path | None = None,
):
    torch.manual_seed(cfg.test.seed)
    if cfg.train.device == "cuda":
        torch.cuda.manual_seed(cfg.test.seed)

    ckpt_path = Path(checkpoint_path or cfg.test.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=cfg.train.device)
    model_cfg = model_config_from_dict(ckpt["model_config"])
    model = create_model(model_cfg).to(cfg.train.device)
    model.load_state_dict(ckpt["model"])

    if cfg.train.dtype == "float16":
        ptdtype = torch.float16
    elif cfg.train.dtype == "bfloat16":
        ptdtype = torch.bfloat16
    else:
        ptdtype = torch.float32
    ctx = (
        torch.amp.autocast(device_type=cfg.train.device, dtype=ptdtype)
        if cfg.train.device == "cuda"
        else nullcontext()
    )

    test_data = BinDataset(
        (
            Path(dataset_dir)
            if dataset_dir is not None
            else cfg.dataset.resolved_processed_dir()
        )
        / "test.bin",
        model_cfg.block_size,
        cfg.train.device,
    )
    loss_metrics = evaluate_test_loss(model, test_data, cfg, ctx)
    prompts = load_prompts_from_test_bin(
        cfg,
        dataset_dir=dataset_dir,
        block_size=model_cfg.block_size,
    )
    samples = generate_samples(model, cfg, prompts)

    results = {
        "checkpoint": str(ckpt_path),
        "iter_num": ckpt.get("iter_num"),
        "best_valid_loss": ckpt.get("best_valid_loss"),
        **loss_metrics,
        "samples": samples,
    }

    results_path = Path(results_path or cfg.test.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v for k, v in results.items() if k != "samples"}, indent=2))
    print(f"Saved test results to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyStories GPT.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Processed dataset directory containing test.bin.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    cfg = deepcopy(CONFIG)
    if args.device is not None:
        cfg.train.device = args.device

    test(
        cfg,
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint,
        results_path=args.results_path,
    )


if __name__ == "__main__":
    main()
