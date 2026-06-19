"""Download TinyStories and prepare GPT-2-tokenized train/valid/test memmaps."""

import argparse
from copy import deepcopy
import json
import random

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from src.config import CONFIG, DatasetConfig, max_stories_label


def _validate_split(cfg: DatasetConfig):
    total = cfg.train_pct + cfg.valid_pct + cfg.test_pct
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_pct + valid_pct + test_pct must equal 1.0, got {total}"
        )


def prepare_dataset(cfg: DatasetConfig = CONFIG.dataset):
    _validate_split(cfg)
    random.seed(cfg.seed)

    processed_dir = cfg.resolved_processed_dir()
    processed_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(cfg.tokenizer_name)
    eot = enc.eot_token

    print(f"Loading dataset: {cfg.dataset_name}")
    ds = load_dataset(cfg.dataset_name, split="train")

    if cfg.max_stories is not None:
        ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.max_stories, len(ds))))
    else:
        ds = ds.shuffle(seed=cfg.seed)

    n = len(ds)
    n_train = int(n * cfg.train_pct)
    n_valid = int(n * cfg.valid_pct)
    n_test = n - n_train - n_valid

    splits = {
        "train": ds.select(range(0, n_train)),
        "valid": ds.select(range(n_train, n_train + n_valid)),
        "test": ds.select(range(n_train + n_valid, n)),
    }

    metadata = {
        "dataset_name": cfg.dataset_name,
        "tokenizer_name": cfg.tokenizer_name,
        "vocab_size": enc.n_vocab,
        "eot_token": eot,
        "max_stories": cfg.max_stories,
        "max_stories_label": max_stories_label(cfg.max_stories),
        "processed_dir": str(processed_dir),
        "num_stories": n,
        "splits": {"train": n_train, "valid": n_valid, "test": n_test},
        "block_size": cfg.block_size,
    }

    for split_name, split_ds in splits.items():
        token_ids = []
        print(f"Tokenizing {split_name}: {len(split_ds)} stories")
        for row in tqdm(split_ds, desc=f"{split_name}"):
            text = row[cfg.text_column]
            ids = enc.encode_ordinary(text)
            token_ids.extend(ids + [eot])

        arr = np.array(token_ids, dtype=np.uint16)
        out_path = processed_dir / f"{split_name}.bin"
        arr.tofile(out_path)
        metadata[f"{split_name}_tokens"] = int(arr.size)
        print(f"Wrote {out_path} with {arr.size:,} tokens")

    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Dataset preparation complete.")
    print(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyStories token bins.")
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Number of stories to process. Defaults to config value.",
    )
    parser.add_argument(
        "--all-stories",
        action="store_true",
        help="Use all available TinyStories rows.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Base output directory. The max_stories label is appended.",
    )
    args = parser.parse_args()

    cfg = deepcopy(CONFIG.dataset)
    if args.all_stories:
        cfg.max_stories = None
    elif args.max_stories is not None:
        cfg.max_stories = args.max_stories
    if args.processed_dir is not None:
        cfg.processed_dir = args.processed_dir

    prepare_dataset(cfg)


if __name__ == "__main__":
    main()
