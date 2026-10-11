"""Publish a REAL, evaluated FlashHAR-X checkpoint to Hugging Face.

Usage:
  hf auth login
  python publish_hf.py --repo-id YOUR_HF_USERNAME/FlashHAR-X
  python publish_hf.py --repo-id YOUR_HF_USERNAME/FlashHAR-X --private

Never upload data/, local tokens, raw .pt pickle checkpoints or unverified scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]
CHANNELS = [f"{prefix}_{axis}" for prefix in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"]


def model_card(repo_id: str, accuracy: float, macro_f1: float, rows: list[dict], github_repo: str | None = None) -> str:
    table = "\n".join(
        f"| {row['model']} | {float(row['accuracy']):.4f} | {float(row['macro_f1']):.4f} |"
        for row in rows
    )
    source = f"- **GitHub source:** https://github.com/{github_repo}\n" if github_repo else ""
    return f"""---
library_name: pytorch
tags:
- human-activity-recognition
- time-series-classification
- sensor-fusion
- flash-attention
- triton
metrics:
- accuracy
- f1
---

# FlashHAR-X: Subject-Generalizable Human Activity Recognition

**Trained model** (not a Transformers checkpoint). Custom multiscale 1D-convolution + two-block attention model for classifying smartphone inertial signals.

{source}- **Official subject-disjoint UCI HAR test accuracy:** **{accuracy:.4f}** ({accuracy*100:.2f}%)
- **Official test macro-F1:** **{macro_f1:.4f}**
- **Task:** 6-class human activity recognition.
- **Dataset:** [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K). The original dataset is CC BY 4.0. **No dataset copies are included in this model repository.**
- **Training:** official train subjects (7,352 windows); validation subject groups selected within training to fix epoch counts, then retrained using all official training subjects. Official held-out test: 2,947 windows from distinct subjects. Scores use the standard published split, not random window splits.
- **Inputs:** one 2.56-second sensor window containing 9 channels × 128 readings **plus** the original dataset's 561 engineered features, in the same order and normalization as training. Do not feed arbitrary accelerometer streams without replicating UCI preprocessing.
- **Architecture:** 3/7/15 multiscale convolution stem → positional encoding → two attention blocks → mean/max pooling → engineered-feature MLP fusion → 6-class classifier.
- **Attention:** trained using a *custom, educational Triton FlashAttention-style forward and backward kernel*. For portability, inference can use numerically equivalent PyTorch `scaled_dot_product_attention` with the SAME learned weights, including on CPU. This isn't a claim that the custom kernel runs faster than PyTorch SDPA.

## Official test results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
{table}

These results are produced by this project's `artifacts/official_test_metrics.csv` after completing the notebook, not copied from papers. Other research numbers are only comparable when inputs, split, and metric definitions match. The checkpoint chosen here is the **custom-Triton-trained** model, *not necessarily the highest-scoring baseline*.

## Files

- `model.safetensors`: trained neural network parameters (no pickle).
- `config.json`: architecture details and class ordering.
- `normalization.json`: means/scales fitted on official train subjects only.
- `official_test_metrics.csv`: reported benchmark numbers.
- `flashhar_model.py`: network definition.

## Reproduce predictions

Clone the associated GitHub project and install its inference dependencies, then:

```bash
python inference_hf.py --repo-id {repo_id} --dataset-dir data --index 0 --device cpu
```

The notebook must first have downloaded/extracted the official UCI HAR dataset under `data/`. The inference script reads a **real** official test window. Install `torch`, `numpy`, `huggingface_hub` and `safetensors` to run inference; running the custom Triton backend for training requires an appropriately configured CUDA GPU.

## Important limitations

- Subject-independent testing on 30 volunteers does not demonstrate reliability for every user, device orientation, sensor vendor, mobility pattern, or unconstrained real-world scenario.
- The model uses UCI's **precomputed** 561 engineered features in addition to raw inertial signals. Real-time deployment requires reproducing those features, not just forwarding nine sensor channels.
- The notebook's GPU experiments should be independently verified; benchmark numbers and any performance claims belong to their recorded hardware and seed.
- This is a college research prototype, not a safety-critical or medical system.

## References

Reyes-Ortiz et al., *Human Activity Recognition Using Smartphones* (2013). FlashAttention: Dao et al. (2022); FlashAttention-2: Dao (2023). Our Triton implementation is an educational reimplementation of tiled exact attention, not a novel attention algorithm.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. username/FlashHAR-X")
    parser.add_argument("--private", action="store_true", help="Publish a private HF model repository")
    parser.add_argument("--github-repo", help="Optional OWNER/REPO link for model card")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/triton_flash.pt")
    parser.add_argument("--metrics", type=Path, default=ROOT / "artifacts/official_test_metrics.csv")
    args = parser.parse_args()

    if args.github_repo and (len(args.github_repo.split("/")) != 2 or not all(args.github_repo.split("/"))):
        parser.error("--github-repo must be OWNER/REPO")
    if len(args.repo_id.split("/")) != 2 or not all(args.repo_id.split("/")):
        parser.error("--repo-id must be username-or-org/model-name")
    if not args.checkpoint.is_file() or not args.metrics.is_file():
        parser.error("Missing trained checkpoint or official test metrics. Complete the notebook's final training and evaluation cells before publishing.")

    with args.metrics.open(newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    matches = [r for r in rows if r["model"] == "triton_flash"]
    if len(matches) != 1:
        parser.error("Metrics file must contain exactly one 'triton_flash' model result.")
    accuracy, macro_f1 = float(matches[0]["accuracy"]), float(matches[0]["macro_f1"])
    if not (0 <= accuracy <= 1 and 0 <= macro_f1 <= 1):
        parser.error("Invalid test metrics; refusing to publish.")

    # Only deserialize YOUR OWN locally-generated notebook checkpoint, never an untrusted .pt file.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("backend") != "triton_flash" or not isinstance(ckpt.get("model"), dict):
        parser.error("Expected a notebook-generated triton_flash checkpoint.")
    normal = ckpt["normalization"]
    preprocessing = {
        "raw_mean": np.asarray(normal["raw_mean"], dtype=np.float32).tolist(),
        "raw_std": np.asarray(normal["raw_std"], dtype=np.float32).tolist(),
        "features_mean": np.asarray(normal["features_mean"], dtype=np.float32).tolist(),
        "features_scale": np.asarray(normal["features_scale"], dtype=np.float32).tolist(),
    }
    if np.asarray(preprocessing["raw_mean"]).shape != (1, 9, 1):
        parser.error("Unexpected raw mean shape; check notebook preprocessing.")
    if np.asarray(preprocessing["features_mean"]).shape != (561,):
        parser.error("Unexpected engineered feature statistics shape.")

    config = {
        "architecture": "FlashHARX", "width": 128, "depth": 2, "multiscale": True,
        "use_features": bool(ckpt.get("use_features", True)),
        "trained_attention_backend": "triton_flash",
        "supported_inference_backend": "torch_sdpa",
        "num_labels": 6, "id2label": {str(i): label for i, label in enumerate(ACTIVITIES)},
        "raw_channels": CHANNELS, "raw_shape": [9, 128], "feature_dim": 561,
        "training_seed": int(ckpt["seed"]), "selected_epochs": int(ckpt["epochs"]),
        "test_accuracy": accuracy, "test_macro_f1": macro_f1,
        "dataset": "UCI HAR Using Smartphones; DOI 10.24432/C54S4K",
    }
    tensors = {k: v.detach().contiguous().cpu() for k, v in ckpt["model"].items()}
    api = HfApi()
    user = api.whoami()
    print("Authenticated to Hugging Face as:", user.get("name", "(unknown)"))
    print("Publishing:", args.repo_id, "(private)" if args.private else "(public)")
    with tempfile.TemporaryDirectory(prefix="flashhar_release_") as tmp:
        release = Path(tmp)
        save_file(tensors, release / "model.safetensors")
        (release / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf8")
        (release / "normalization.json").write_text(json.dumps(preprocessing, indent=2) + "\n", encoding="utf8")
        (release / "README.md").write_text(model_card(args.repo_id, accuracy, macro_f1, rows, args.github_repo), encoding="utf8")
        shutil.copy2(args.metrics, release / "official_test_metrics.csv")
        shutil.copy2(ROOT / "flashhar_model.py", release / "flashhar_model.py")
        for name in ("confusion_matrix.png", "experiment_manifest.json"):
            file = args.checkpoint.parent / name
            if file.is_file(): shutil.copy2(file, release / name)
        sha256 = hashlib.sha256((release / "model.safetensors").read_bytes()).hexdigest()
        (release / "SHA256SUMS.txt").write_text(f"{sha256}  model.safetensors\n", encoding="utf8")
        print("Uploading", sorted(p.name for p in release.iterdir()))
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(release), commit_message="Release evaluated FlashHAR-X trained checkpoint and model card")
    print("Published: https://huggingface.co/" + args.repo_id)


if __name__ == "__main__":
    main()
