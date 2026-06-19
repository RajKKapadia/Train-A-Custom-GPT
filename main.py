"""Run the full TinyStories GPT process end-to-end."""

import argparse
from copy import deepcopy

from src.config import CONFIG
from src.prepare_dataset import prepare_dataset
from src.test import test
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Processed dataset directory to use for training/testing.",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Override the configured max_stories before preparing data.",
    )
    parser.add_argument(
        "--all-stories",
        action="store_true",
        help="Prepare data using all available stories.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Short description for the training experiment.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Base directory where timestamped experiment folders are created.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep ckpt_*.pt and last.pt after a successful run.",
    )
    args = parser.parse_args()

    if args.all_stories and args.max_stories is not None:
        parser.error("--all-stories cannot be used with --max-stories")

    cfg = deepcopy(CONFIG)
    if args.all_stories:
        cfg.dataset.max_stories = None
    elif args.max_stories is not None:
        cfg.dataset.max_stories = args.max_stories
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.device is not None:
        cfg.train.device = args.device
    if args.keep_checkpoints:
        cfg.train.delete_intermediate_checkpoints = False
    cfg.sync()

    dataset_dir = args.dataset_dir

    if not args.skip_prepare:
        metadata = prepare_dataset(cfg.dataset)
        if dataset_dir is None:
            dataset_dir = metadata["processed_dir"]

    if not args.skip_train:
        train_result = train(
            cfg,
            dataset_dir=str(dataset_dir),
            description=args.description,
        )
        dataset_dir = str(train_result.dataset_dir)
        if args.checkpoint is None:
            cfg.test.checkpoint_path = str(train_result.best_checkpoint_path)
            cfg.test.results_path = str(train_result.out_dir / "test_results.json")

    if not args.skip_test:
        test(
            cfg,
            dataset_dir=str(dataset_dir),
            checkpoint_path=args.checkpoint,
        )


if __name__ == "__main__":
    main()
