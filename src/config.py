"""Central configuration for TinyStories GPT training."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json


def max_stories_label(max_stories: int | None) -> str:
    if max_stories is None:
        return "max_stories_all"
    return f"max_stories_{max_stories}"


def processed_dataset_dir(processed_dir: str | Path, max_stories: int | None) -> Path:
    base_dir = Path(processed_dir)
    label = max_stories_label(max_stories)
    if base_dir.name == label:
        return base_dir
    return base_dir / label


@dataclass
class DatasetConfig:
    dataset_name: str = "roneneldan/TinyStories"
    text_column: str = "text"

    # Keep this small for your first local run. Set to None to use all loaded rows.
    # TinyStories is large, so start with 50k/100k stories first.
    max_stories: int | None = 50_000

    train_pct: float = 0.90
    valid_pct: float = 0.05
    test_pct: float = 0.05

    tokenizer_name: str = "gpt2"
    block_size: int = 256
    seed: int = 1337

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    def resolved_processed_dir(self) -> Path:
        return processed_dataset_dir(self.processed_dir, self.max_stories)


@dataclass
class ModelConfig:
    vocab_size: int = 50_257  # GPT-2 tokenizer vocab size
    block_size: int = 512
    n_layer: int = 10
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.10
    bias: bool = True


@dataclass
class TrainConfig:
    out_dir: str = "runs/tinystories-gpt"
    experiment_description: str = ""
    delete_intermediate_checkpoints: bool = True
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_iters: int = 10_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    warmup_iters: int = 500
    lr_decay_iters: int = 10_000
    min_lr: float = 3e-5

    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 20
    save_interval: int = 1000

    device: str = "cuda"  # use "cpu" only for debugging
    dtype: str = "float16"  # "float16", "bfloat16", or "float32"
    compile_model: bool = False
    seed: int = 1337


@dataclass
class TestConfig:
    checkpoint_path: str = "runs/tinystories-gpt/best.pt"
    results_path: str = "runs/tinystories-gpt/test_results.json"
    num_eval_batches: int = 200
    num_generation_prompts: int = 20
    max_new_tokens: int = 300
    temperature: float = 0.7
    top_k: int | None = 50
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 4
    seed: int = 1337


@dataclass
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)

    def sync(self) -> "AppConfig":
        self.model.block_size = self.dataset.block_size
        return self

    def to_dict(self):
        return asdict(self)

    def save_json(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


CONFIG = AppConfig().sync()
