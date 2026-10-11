# FlashHAR-X — Real Sensor Recognition with Custom FlashAttention

A subject-independent Human Activity Recognition project using the **official 2013 UCI HAR smartphone dataset**, custom Triton FlashAttention-style forward **and** backward kernels, benchmarked against PyTorch SDPA, a CNN baseline, and a feature-based RBF-SVM.

## Notebook-first quickstart — WSL Ubuntu + RTX 5070 Ti

```bash
cd ~/projects/FlashHAR_X  # copy the entire folder here
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m ipykernel install --user --name flashhar-x --display-name 'Python (FlashHAR-X)'
jupyter lab
```

Open `FlashHAR_X_Train_and_Benchmark.ipynb`, select the FlashHAR-X kernel, **Run All**. Dataset downloaded from `https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip` (~58MB). If offline, download the archive manually from https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones and place it in `data/`. The download cell unpacks nested zips.

Before starting, run `nvidia-smi` inside WSL to check the GPU is visible. Do not use native Windows Python for this Triton kernel. No `flash-attn` pip installation or CUDA C++ compiler is required by the handwritten Triton implementation (Triton JIT is used); a supported GPU driver is required.

## Files

- `FlashHAR_X_Train_and_Benchmark.ipynb`: explanatory runnable notebook; downloads authentic data, trains models, validates, evaluates test once, benchmarks and saves artifacts.
- `flash_attention_triton.py`: own tiled FP16 FlashAttention-style online softmax forward and recomputation backward kernels. Not a PyTorch SDPA wrapper. **Educational prototype, not a production-performance implementation.**
- `flashhar_model.py`: reusable model, attention backend ablation, sensor augmentation.
- `requirements.txt`: Python dependencies (install CUDA PyTorch separately).
- `artifacts/`: created automatically (checkpoints, reports, confusion matrix, timing).

## Research protocol

Official **7352 training windows** / **2947 test windows**. The groups are volunteer IDs, never random overlapping windows. GroupShuffleSplit creates train/validation subject separation *within the official training subjects*. All preprocessing for selection is fitted on training subjects only. The model epoch count is chosen using macro F1 on validation volunteers. The model is then refit on all the official training people for that count, and evaluated on test volunteers once.

Data includes 9 real inertial signals x 128 readings + optionally the 561 feature vector computed by UCI from the original windows. As such, SVM on precomputed features is **not input-equivalent** to an IMU-only model. If using both signal and feature branches, noise added to signal alone leaves the feature branch clean, so do not call it full-sensor robustness.

**Targets:** >=90% accuracy on official unseen-subject test, stretch >=97%. These are goals, not preobserved results. Reports will print when the goal is not achieved. Any comparison to published papers requires verifying equal dataset version, channels, subject split, preprocessing and accuracy-vs-macro-F1 definition.

### Engineering notes

- Custom Triton kernel operates on CUDA FP16, contiguous `[batch, heads, sequence, dim]`, sequence divisible by 64, `head_dim=32/64`, noncausal full attention without attention dropout or bias. Gradients are rederived with tile recomputation; no dense NxN probability tensor.
- Forward and all Q/K/V gradients must match PyTorch SDPA within the notebook's numeric tolerances before training starts.
- PyTorch SDPA is an optimized GPU backend, not an unfused mathematical reference. At **128 samples**, the handwritten kernel may be slower than SDPA. Do not claim speed gains without actual CUDA timings and memory profiling.
- No synthetic activity recordings are produced. Random tensors in kernel numerical tests/microbenchmark are **not training records**.
- Using overlapping windows within each person is inherent to the official protocol; preserving subject-disjoint splits avoids duplicate-person leakage, but this benchmark does not prove geographic/real-world deployment generalization.

## Useful outputs

`artifacts/official_test_metrics.csv`, `artifacts/experiment_manifest.json`, `artifacts/confusion_matrix.png`, `artifacts/flash_attention_forward_benchmark.csv`, model `.pt` files.

## Attribution and references

Dataset: Reyes-Ortiz et al., *Human Activity Recognition Using Smartphones* (2013), UCI Machine Learning Repository, [DOI 10.24432/C54S4K](https://doi.org/10.24432/C54S4K), license CC BY 4.0.

FlashAttention: [Dao et al. 2022](https://arxiv.org/abs/2205.14135); FlashAttention-2: [Dao 2023](https://arxiv.org/abs/2307.08691). Custom Triton implementation inspired by the architecture and teaching explanations in [Triton's official fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html). Not claiming FlashAttention as a novel algorithm. 
