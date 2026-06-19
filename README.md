# TinyStories GPT from Scratch

A small decoder-only GPT training project using TinyStories and GPT-2 `tiktoken` tokenization. It supports training/evaluation in PyTorch, cached PyTorch generation, and forward-only ONNX Runtime generation.

## Files

- `src/config.py` - model, dataset, train, and generation configuration
- `src/prepare_dataset.py` - downloads TinyStories, tokenizes with GPT-2 tokenizer, creates train/valid/test `.bin` files
- `src/create_model.py` - builds the GPT model, including KV-cache generation and sampling controls
- `src/train.py` - trains, validates, saves checkpoints and metrics
- `src/test.py` - evaluates test loss/perplexity and saves generated samples
- `main.py` - runs prepare/train/test end to end
- `generate.py` - generates text from a PyTorch checkpoint
- `evaluate_generation.py` - evaluates prompts and saves generation metrics/samples
- `export_onnx.py` - exports a checkpoint to ONNX Runtime format
- `generate_onnx.py` - generates text from an exported ONNX model

## Setup

```bash
uv sync
```

If using plain Python instead of `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run End To End

```bash
uv run python main.py \
  --description "baseline 50k stories"
```

Useful options:

```bash
uv run python main.py --skip-prepare
uv run python main.py --skip-train --checkpoint runs/tinystories-gpt/<experiment-folder>/best.pt
uv run python main.py --device cpu
uv run python main.py --keep-checkpoints
```

## Run Step By Step

```bash
uv run python src/prepare_dataset.py --max-stories 50000

uv run python src/train.py \
  --dataset-dir data/processed/max_stories_50000 \
  --description "baseline 50k stories"

uv run python src/test.py \
  --dataset-dir data/processed/max_stories_50000 \
  --checkpoint runs/tinystories-gpt/<experiment-folder>/best.pt
```

Prepared datasets are written under a max-stories-labeled folder:

```text
data/processed/max_stories_50000/train.bin
data/processed/max_stories_50000/valid.bin
data/processed/max_stories_50000/test.bin
data/processed/max_stories_50000/metadata.json
```

Training creates timestamped experiment folders:

```text
runs/tinystories-gpt/yyyy_mm_dd_hh_mm/
```

Each experiment folder includes `config.json`, `experiment.json`, metrics files, and `best.pt`. After a successful run, intermediate `ckpt_*.pt` files and `last.pt` are deleted by default. Use `--keep-checkpoints` to retain them.

## PyTorch Generation

Generate from a checkpoint:

```bash
uv run python generate.py \
  --checkpoint runs/tinystories-gpt/<experiment-folder>/best.pt \
  --prompt "Once upon a time"
```

Generation uses KV cache by default. To compare against the slower full-context path:

```bash
uv run python generate.py \
  --checkpoint runs/tinystories-gpt/<experiment-folder>/best.pt \
  --prompt "Once upon a time" \
  --no-kv-cache
```

Current default sampling controls are:

```text
temperature: 0.7
top_k: 50
repetition_penalty: 1.05
no_repeat_ngram_size: 4
max_new_tokens: 300
```

For deterministic greedy generation:

```bash
uv run python generate.py \
  --checkpoint runs/tinystories-gpt/<experiment-folder>/best.pt \
  --prompt "Once upon a time" \
  --temperature 0 \
  --top-k 0 \
  --repetition-penalty 1 \
  --no-repeat-ngram-size 0
```

## Generation Evaluation

Evaluate the prompts in `eval_prompts.txt`:

```bash
uv run python evaluate_generation.py \
  --checkpoint runs/tinystories-gpt/<experiment-folder>/best.pt
```

This writes:

```text
runs/tinystories-gpt/<experiment-folder>/generation_eval/generation_eval.json
runs/tinystories-gpt/<experiment-folder>/generation_eval/generation_eval.csv
runs/tinystories-gpt/<experiment-folder>/generation_eval/generation_samples.txt
```

## ONNX Export

Export a trained checkpoint:

```bash
uv run python export_onnx.py \
  --checkpoint runs/tinystories-gpt/<experiment-folder>/best.pt \
  --output runs/tinystories-gpt/<experiment-folder>/model.onnx
```

This writes:

```text
runs/tinystories-gpt/<experiment-folder>/model.onnx
runs/tinystories-gpt/<experiment-folder>/model.onnx.json
```

The exporter validates ONNX logits against PyTorch logits. A small max absolute difference, for example around `1e-5`, is expected.

You may see PyTorch ONNX exporter warnings about the legacy exporter or tracing Python conditionals. The exported model is valid for normal generation because `generate_onnx.py` crops input length to the configured `block_size`.

## ONNX Runtime Generation

Generate with ONNX Runtime:

```bash
uv run python generate_onnx.py \
  --onnx runs/tinystories-gpt/<experiment-folder>/model.onnx \
  --prompt "Once upon a time"
```

Deterministic ONNX generation for comparing with PyTorch:

```bash
uv run python generate_onnx.py \
  --onnx runs/tinystories-gpt/<experiment-folder>/model.onnx \
  --prompt "Once upon a time" \
  --max-new-tokens 16 \
  --temperature 0 \
  --top-k 0 \
  --repetition-penalty 1 \
  --no-repeat-ngram-size 0
```

The current ONNX graph is forward-only: it returns last-token logits for a cropped context and recomputes that context each generated token. The PyTorch path has KV-cache generation; a cache-aware ONNX export would require a separate graph with cache tensors as explicit inputs and outputs.

## First-Run Advice

Start with `max_stories = 50_000` in `src/config.py` or pass `--max-stories 50000`. Once the pipeline works, increase it to `100_000`, `500_000`, or use `--all-stories`.

For a 16 GB GPU, the default config is intentionally moderate:

```text
layers: 10
heads: 8
embedding: 512
context: 256
batch_size: 16
gradient_accumulation_steps: 4
```

Effective tokens per optimization step:

```text
batch_size x gradient_accumulation_steps x block_size
= 16 x 4 x 256
= 16,384 tokens
```

## Outputs

After training:

```text
runs/tinystories-gpt/yyyy_mm_dd_hh_mm/config.json
runs/tinystories-gpt/yyyy_mm_dd_hh_mm/experiment.json
runs/tinystories-gpt/yyyy_mm_dd_hh_mm/metrics.csv
runs/tinystories-gpt/yyyy_mm_dd_hh_mm/metrics.jsonl
runs/tinystories-gpt/yyyy_mm_dd_hh_mm/best.pt
runs/tinystories-gpt/yyyy_mm_dd_hh_mm/test_results.json
```

Generated ONNX artifacts live in the same experiment folder and are ignored by git because `runs/` is ignored.

## Notes

This is plain next-token pretraining. It is not instruction tuning yet. TinyStories is useful for checking whether the model learns grammar, short-story structure, and coherent continuation.
