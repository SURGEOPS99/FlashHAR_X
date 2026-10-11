"""Verify a published FlashHAR-X checkpoint on an actual UCI HAR test recording.

Example:
  python inference_hf.py --repo-id USERNAME/FlashHAR-X --dataset-dir data --index 0 --device cpu

This project is NOT a Hugging Face Transformers model; it loads native PyTorch weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from flashhar_model import FlashHARX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but PyTorch cannot see the GPU")
    if args.index < 0:
        parser.error("--index cannot be negative")

    paths = {name: Path(hf_hub_download(repo_id=args.repo_id, filename=name)) for name in
             ("model.safetensors", "config.json", "normalization.json")}
    cfg = json.loads(paths["config.json"].read_text(encoding="utf8"))
    norm = json.loads(paths["normalization.json"].read_text(encoding="utf8"))
    model = FlashHARX(backend="torch_sdpa", width=cfg["width"], depth=cfg["depth"],
                      use_features=cfg["use_features"], multiscale=cfg["multiscale"])
    model.load_state_dict(load_file(str(paths["model.safetensors"])), strict=True)
    model = model.to(args.device).eval()

    # Read a REAL UCI window, preserving the notebook's exact channel ordering.
    candidates = list(args.dataset_dir.rglob("body_acc_x_test.txt"))
    if len(candidates) != 1:
        parser.error("Expected exactly one extracted UCI HAR dataset under --dataset-dir. Run notebook download cell first.")
    signals = candidates[0].parent
    test_dir = signals.parent
    if not (test_dir / "X_test.txt").is_file():
        parser.error("Missing X_test.txt alongside UCI inertial signals")
    raw = np.stack([
        np.loadtxt(signals / f"{channel}_test.txt", dtype=np.float32, max_rows=args.index + 1, ndmin=2)[args.index]
        for channel in cfg["raw_channels"]
    ], axis=0)[None, :, :]
    feat = np.loadtxt(test_dir / "X_test.txt", dtype=np.float32, max_rows=args.index + 1, ndmin=2)[args.index:args.index+1]
    if raw.shape != (1, 9, 128) or feat.shape != (1, 561):
        raise ValueError(f"Incorrect signal/feature shape: {raw.shape}, {feat.shape}")
    raw = (raw - np.asarray(norm["raw_mean"], dtype=np.float32)) / np.asarray(norm["raw_std"], dtype=np.float32)
    feat = (feat - np.asarray(norm["features_mean"], dtype=np.float32)) / np.asarray(norm["features_scale"], dtype=np.float32)
    raw_t = torch.from_numpy(raw.astype(np.float32)).to(args.device)
    feat_t = torch.from_numpy(feat.astype(np.float32)).to(args.device)
    with torch.inference_mode():
        scores = torch.softmax(model(raw_t, feat_t), dim=-1)[0].cpu().numpy()
    labels = [cfg["id2label"][str(i)] for i in range(cfg["num_labels"])]
    print("Predicted:", labels[int(scores.argmax())])
    for label, score in sorted(zip(labels, scores), key=lambda item: -item[1]):
        print(f"  {label:<20s} {score:.4f}")
    labels_file = test_dir / "y_test.txt"
    if labels_file.is_file():
        truth = int(np.loadtxt(labels_file, dtype=np.int64, max_rows=args.index + 1, ndmin=1)[-1])
        print("Actual label:", labels[truth - 1])


if __name__ == "__main__":
    main()
