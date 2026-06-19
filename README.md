# TinyStories GPT from Scratch

A small decoder-only GPT training project using TinyStories and GPT-2 `tiktoken` tokenization.

## Files

- `src/config.py` — model, dataset, train, and test configuration
- `src/prepare_dataset.py` — downloads TinyStories, tokenizes with GPT-2 tokenizer, creates train/valid/test `.bin` files
- `src/create_model.py` — builds a decoder-only GPT model from config
- `src/train.py` — trains, validates, saves checkpoints and metrics
- `src/test.py` — evaluates test loss/perplexity and saves generated samples
- `main.py` — runs the process end-to-end
- `generate.py` — generates text from a saved checkpoint
- `export_onnx.py` — exports a checkpoint to ONNX Runtime format
- `generate_onnx.py` — generates text from an exported ONNX model

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run end-to-end

```bash
python main.py --description "baseline 50k stories"
```

## Run step-by-step

```bash
python prepare_dataset.py --max-stories 50000
python train.py \
  --dataset-dir data/processed/max_stories_50000 \
  --description "baseline 50k stories"
python test.py \
  --dataset-dir data/processed/max_stories_50000 \
  --checkpoint runs/tinystories-gpt/<experiment-folder>/best.pt
```

Prepared datasets are written under a max-stories-labeled folder, for example:

```text
data/processed/max_stories_50000/train.bin
data/processed/max_stories_50000/valid.bin
data/processed/max_stories_50000/test.bin
data/processed/max_stories_50000/metadata.json
```

Training creates timestamped experiment folders with the format:

```text
runs/tinystories-gpt/yyyy_mm_dd_hh_mm/
```

Each experiment folder includes `config.json`, `experiment.json`, metrics files, and `best.pt`. The experiment description is saved in `experiment.json`. After a successful run, intermediate `ckpt_*.pt` files and `last.pt` are deleted by default. Use `--keep-checkpoints` if you want to retain them.

## ONNX export and generation

Export a trained checkpoint:

```bash
python export_onnx.py \
  --checkpoint runs/tinystories-gpt/<experiment-folder>/best.pt
```

This writes `model.onnx` and `model.onnx.json` in the checkpoint folder. Generate with ONNX Runtime:

```bash
python generate_onnx.py \
  --onnx runs/tinystories-gpt/<experiment-folder>/model.onnx \
  --prompt "Once upon a time"
```

## First-run advice

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
batch_size × gradient_accumulation_steps × block_size
= 16 × 4 × 256
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

## Notes

This is plain next-token pretraining. It is not instruction tuning yet. TinyStories is good for learning whether the model can learn grammar, short stories, and coherent continuation.
