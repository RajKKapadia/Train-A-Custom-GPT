"""Train and validate the TinyStories GPT model."""

import argparse
from pathlib import Path
import csv
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import math
import time
from typing import cast

import numpy as np
import torch

from src.config import CONFIG, AppConfig, TrainConfig
from src.create_model import GPT, create_model


@dataclass
class TrainResult:
    out_dir: Path
    best_checkpoint_path: Path
    dataset_dir: Path
    experiment_description: str


def get_lr(iter_num: int, cfg: TrainConfig):
    if iter_num < cfg.warmup_iters:
        return cfg.learning_rate * iter_num / max(1, cfg.warmup_iters)
    if iter_num > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (iter_num - cfg.warmup_iters) / max(
        1, cfg.lr_decay_iters - cfg.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def _make_experiment_dir(base_dir: Path) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_name = timestamp
    candidate = base_dir / run_name
    if not candidate.exists():
        return candidate, timestamp

    for suffix in range(2, 1000):
        candidate = base_dir / f"{run_name}_{suffix}"
        if not candidate.exists():
            return candidate, timestamp

    raise RuntimeError(
        f"Could not create a unique experiment directory under {base_dir}"
    )


def _validate_dataset_dir(dataset_dir: Path):
    missing = [
        path
        for path in (
            dataset_dir / "train.bin",
            dataset_dir / "valid.bin",
        )
        if not path.exists()
    ]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Dataset directory is missing required files: {missing_paths}"
        )


def cleanup_intermediate_checkpoints(out_dir: Path) -> list[Path]:
    best_path = out_dir / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint was not found, refusing cleanup: {best_path}"
        )

    deleted = []
    for path in sorted(out_dir.glob("ckpt_*.pt")):
        path.unlink()
        deleted.append(path)

    last_path = out_dir / "last.pt"
    if last_path.exists():
        last_path.unlink()
        deleted.append(last_path)

    return deleted


class BinDataset:
    def __init__(self, path: str | Path, block_size: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.device = device

    def get_batch(self, batch_size: int):
        ix = torch.randint(len(self.data) - self.block_size - 1, (batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y


def make_optimizer(model, cfg: TrainConfig):
    """Create AdamW optimizer groups.

    This works correctly even when token embedding and lm_head weights are tied.
    """

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Matrices like Linear/Embedding weights get weight decay.
        # Biases and LayerNorm vectors do not.
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        optim_groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )


@torch.no_grad()
def estimate_loss(model, train_data, valid_data, cfg: AppConfig, ctx):
    out = {}
    model.eval()
    for split_name, dataset in [("train", train_data), ("valid", valid_data)]:
        losses = torch.zeros(cfg.train.eval_iters)
        for k in range(cfg.train.eval_iters):
            X, Y = dataset.get_batch(cfg.train.batch_size)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


def train(
    cfg: AppConfig = CONFIG,
    dataset_dir: str | Path | None = None,
    description: str | None = None,
) -> TrainResult:
    torch.manual_seed(cfg.train.seed)
    if cfg.train.device == "cuda":
        torch.cuda.manual_seed(cfg.train.seed)

    processed_dir = (
        Path(dataset_dir)
        if dataset_dir is not None
        else cfg.dataset.resolved_processed_dir()
    )
    _validate_dataset_dir(processed_dir)

    experiment_description = (
        description if description is not None else cfg.train.experiment_description
    )
    cfg.train.experiment_description = experiment_description
    if cfg.train.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    out_dir, experiment_timestamp = _make_experiment_dir(Path(cfg.train.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out_dir / "config.json")

    experiment = {
        "timestamp": experiment_timestamp,
        "description": experiment_description,
        "dataset_dir": str(processed_dir),
        "max_stories": cfg.dataset.max_stories,
        "out_dir": str(out_dir),
        "delete_intermediate_checkpoints": cfg.train.delete_intermediate_checkpoints,
    }
    with (out_dir / "experiment.json").open("w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2)

    train_data = BinDataset(
        processed_dir / "train.bin", cfg.model.block_size, cfg.train.device
    )
    valid_data = BinDataset(
        processed_dir / "valid.bin", cfg.model.block_size, cfg.train.device
    )

    model = create_model(cfg.model).to(cfg.train.device)
    forward_model: GPT = model
    if cfg.train.compile_model:
        forward_model = cast(GPT, torch.compile(model))

    optimizer = make_optimizer(model, cfg.train)

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
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.train.device == "cuda" and cfg.train.dtype == "float16"),
    )

    metrics_path = out_dir / "metrics.csv"
    jsonl_path = out_dir / "metrics.jsonl"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter",
                "train_loss",
                "valid_loss",
                "lr",
                "tokens_seen",
                "time_sec",
            ],
        )
        writer.writeheader()

    best_valid_loss = float("inf")
    t0 = time.time()

    for iter_num in range(cfg.train.max_iters + 1):
        lr = get_lr(iter_num, cfg.train)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % cfg.train.eval_interval == 0:
            losses = estimate_loss(forward_model, train_data, valid_data, cfg, ctx)
            tokens_seen = (
                iter_num
                * cfg.train.batch_size
                * cfg.train.gradient_accumulation_steps
                * cfg.model.block_size
            )
            row = {
                "iter": iter_num,
                "train_loss": losses["train"],
                "valid_loss": losses["valid"],
                "lr": lr,
                "tokens_seen": tokens_seen,
                "time_sec": round(time.time() - t0, 2),
            }
            print(row)
            with metrics_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            if losses["valid"] < best_valid_loss:
                best_valid_loss = losses["valid"]
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": cfg.model.__dict__,
                    "iter_num": iter_num,
                    "best_valid_loss": best_valid_loss,
                    "dataset_dir": str(processed_dir),
                    "experiment": experiment,
                }
                torch.save(ckpt, out_dir / "best.pt")
                print(f"Saved best checkpoint to {out_dir / 'best.pt'}")

        if iter_num == cfg.train.max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        current_loss: torch.Tensor | None = None
        for _ in range(cfg.train.gradient_accumulation_steps):
            X, Y = train_data.get_batch(cfg.train.batch_size)
            with ctx:
                _, loss = forward_model(X, Y)
                current_loss = loss.detach()
                loss = loss / cfg.train.gradient_accumulation_steps
            scaler.scale(loss).backward()

        if cfg.train.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if iter_num % cfg.train.log_interval == 0:
            if current_loss is None:
                raise RuntimeError("No training loss was computed")
            print(
                f"iter {iter_num}: loss {current_loss.item():.4f}, lr {lr:.2e}"
            )

        if iter_num > 0 and iter_num % cfg.train.save_interval == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": cfg.model.__dict__,
                "iter_num": iter_num,
                "best_valid_loss": best_valid_loss,
                "dataset_dir": str(processed_dir),
                "experiment": experiment,
            }
            torch.save(ckpt, out_dir / f"ckpt_{iter_num}.pt")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": cfg.model.__dict__,
            "iter_num": cfg.train.max_iters,
            "best_valid_loss": best_valid_loss,
            "dataset_dir": str(processed_dir),
            "experiment": experiment,
        },
        out_dir / "last.pt",
    )

    if cfg.train.delete_intermediate_checkpoints:
        deleted = cleanup_intermediate_checkpoints(out_dir)
        print(f"Deleted {len(deleted)} intermediate checkpoint(s). Kept best.pt.")

    print("Training complete.")
    return TrainResult(
        out_dir=out_dir,
        best_checkpoint_path=out_dir / "best.pt",
        dataset_dir=processed_dir,
        experiment_description=experiment_description,
    )


def main():
    parser = argparse.ArgumentParser(description="Train TinyStories GPT.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Processed dataset directory containing train.bin and valid.bin.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Short description saved in experiment metadata.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Base directory where timestamped experiment folders are created.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep ckpt_*.pt and last.pt after a successful run.",
    )
    args = parser.parse_args()

    cfg = deepcopy(CONFIG)
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.device is not None:
        cfg.train.device = args.device
    if args.max_iters is not None:
        cfg.train.max_iters = args.max_iters
    if args.keep_checkpoints:
        cfg.train.delete_intermediate_checkpoints = False

    result = train(cfg, dataset_dir=args.dataset_dir, description=args.description)
    print(f"Experiment directory: {result.out_dir}")
    print(f"Best checkpoint: {result.best_checkpoint_path}")


if __name__ == "__main__":
    main()
