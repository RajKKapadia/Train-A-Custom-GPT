"""Export a trained TinyStories GPT checkpoint to ONNX."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from src.config import CONFIG
from src.create_model import GPT, model_config_from_dict


class LastTokenLogits(nn.Module):
    def __init__(self, model: GPT):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        return logits[:, -1, :]


def default_output_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "model.onnx"


def load_checkpoint_model(checkpoint_path: Path, device: str) -> tuple[GPT, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = model_config_from_dict(checkpoint["model_config"])
    model = GPT(model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint


def validate_onnx_export(
    model: GPT,
    onnx_path: Path,
    dummy_input: torch.Tensor,
) -> float:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    with torch.no_grad():
        torch_logits, _ = model(dummy_input)
        torch_logits = torch_logits[:, -1, :].detach().cpu().numpy()

    (ort_logits,) = session.run(
        ["logits"],
        {"input_ids": dummy_input.detach().cpu().numpy().astype(np.int64)},
    )
    return float(np.max(np.abs(torch_logits - ort_logits)))


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    opset: int = 18,
    device: str = "cpu",
    validate: bool = True,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path) if output_path is not None else default_output_path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    wrapper = LastTokenLogits(model).eval()

    dummy_seq_len = min(8, model.config.block_size)
    dummy_input = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, dummy_seq_len),
        dtype=torch.long,
        device=device,
    )

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    max_abs_diff = (
        validate_onnx_export(model, output_path, dummy_input) if validate else None
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iter": checkpoint.get("iter_num"),
        "best_valid_loss": checkpoint.get("best_valid_loss"),
        "model_config": checkpoint["model_config"],
        "tokenizer_name": CONFIG.dataset.tokenizer_name,
        "opset": opset,
        "output": str(output_path),
        "output_shape": ["batch", model.config.vocab_size],
        "validation": {
            "max_abs_diff": max_abs_diff,
            "dummy_input_shape": list(dummy_input.shape),
        },
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported ONNX model: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    if max_abs_diff is not None:
        print(f"Max abs diff vs PyTorch: {max_abs_diff:.6g}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TinyStories GPT to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained best.pt checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path. Defaults to checkpoint folder / model.onnx.",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset=args.opset,
        device=args.device,
        validate=not args.skip_validate,
    )


if __name__ == "__main__":
    main()
