# Publish FlashHAR-X on GitHub + Hugging Face

## 0. Prerequisites

Complete all training and **official held-out test** cells in the notebook. The following must exist:

- `artifacts/triton_flash.pt` — evaluated, trained custom-Triton model weights + train-only normalization statistics
- `artifacts/official_test_metrics.csv` — official held-out test scores

No trained checkpoint has been provided with the template project. Do not publish an untrained or fabricated model.

**Important:** Files under `artifacts/` and `data/` stay local and are ignored by Git. The trained model is published separately on the Hugging Face Model Hub in safe tensor format.

## 1. Publish model to Hugging Face

```bash
cd ~/projects/FlashHAR_X
source .venv/bin/activate
pip install -r requirements-publish.txt
hf auth login                      # enter token via prompt, not source code; token needs write permission
hf auth whoami
python publish_hf.py --repo-id YOUR_HF_USERNAME/FlashHAR-X --github-repo SURGEOPS99/FlashHAR_X
```

To publish privately, add `--private`. This script verifies that the checkpoint and test metrics exist, converts the model state to `model.safetensors`, writes `config.json` and `normalization.json`, generates a truthful card from real reported metrics, and uploads only the intended release files. It does not upload the raw UCI dataset or access tokens.

For a new inference validation on a real UCI test window:

```bash
python inference_hf.py --repo-id YOUR_HF_USERNAME/FlashHAR-X --dataset-dir data --index 0 --device cpu
```

`inference_hf.py` requires the official dataset to be extracted under `data/`. If your notebook was run in this project root, it should already be present.

## 2. Publish source on GitHub

Ensure `.gitignore` is present before `git add`. Review every file you intend to publish. The notebook may contain cell outputs (including local paths); clear them first if you do not want to share them:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('FlashHAR_X_Train_and_Benchmark.ipynb')
b = json.loads(p.read_text())
for c in b['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None
p.write_text(json.dumps(b, indent=1, ensure_ascii=False) + '\n')
PY
```

Then install/authenticate GitHub CLI (`gh`) if needed, and run:

```bash
cd ~/projects/FlashHAR_X
gh auth login
gh auth status
git init
git branch -M main
git add .gitignore README.md FlashHAR_X_Train_and_Benchmark.ipynb \
    flash_attention_triton.py flashhar_model.py requirements.txt kernel_gpu_test.py \
    publish_hf.py inference_hf.py requirements-publish.txt PUBLISHING.md
git status --short
git commit -m "Release FlashHAR-X custom Triton FlashAttention capstone"
gh repo create FlashHAR_X --public --source=. --remote=origin --push
```

If the repository already exists, do NOT rerun `gh repo create`. Instead check `git remote -v`, configure its origin as needed, and `git push -u origin main`.

Optional: copy **real** metrics and figures to a tracked `results/` folder after verifying content, then commit them:

```bash
mkdir -p results
cp artifacts/official_test_metrics.csv results/
cp artifacts/confusion_matrix.png results/  # if it exists
cp artifacts/experiment_manifest.json results/  # if it exists
git add results
git commit -m "Add verified held-out evaluation results"
git push
```

**Never commit** a Hugging Face write token, GitHub token, `.env`, local data, checkpoints, notebooks containing credentials, or private information. The `.gitignore` protects common paths but review `git status --short` before committing.

## 3. Add cross-links

GitHub README: add `Model weights: https://huggingface.co/YOUR_HF_USERNAME/FlashHAR-X`.

Hugging Face model README: add `Source: https://github.com/YOUR_GITHUB_USERNAME/FlashHAR_X`.

## 4. Accuracy claims

If the published model's official test accuracy is below 90%, report that result faithfully. Do not claim state-of-the-art based on dissimilar paper protocols. Custom FlashAttention improves the implementation's attention memory strategy, but accuracy must be measured independently. The current model uses 561 engineered features from the UCI dataset in addition to raw channels; it is not yet a plug-and-play live smartphone model.
