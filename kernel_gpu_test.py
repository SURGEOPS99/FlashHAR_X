"""Standalone mandatory kernel validation on real CUDA GPU in WSL."""
import torch
from flash_attention_triton import check_correctness
if not torch.cuda.is_available():
    raise SystemExit('GPU unavailable. Use WSL Ubuntu and CUDA-enabled PyTorch.')
print(torch.cuda.get_device_name())
check_correctness()
