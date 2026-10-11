"""Educational, non-causal FlashAttention-style Triton autograd implementation.

Design: tiled FP16 QK/PV with FP32 online softmax; backward recomputes
probabilities in separate atomics-free dQ and dK/dV kernels. Activations
never store an N x N attention matrix. Inspired by FlashAttention-2 and
Triton's official fused-attention tutorial, written for this project.

Scope deliberately restricted to contiguous [B,H,N,D] FP16 tensors,
N divisible by 64, D=32 or 64, and no masking/attention dropout.
Tests in the notebook compare both forward AND gradients with PyTorch SDPA.
"""
from __future__ import annotations
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd(Q, K, V, O, LSE,
               N: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    qm = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    base = bh * N * D
    q = tl.load(Q + base + qm[:, None] * D + d[None, :])
    m = tl.full((BM,), -float('inf'), tl.float32)
    l = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, D), 0.0, tl.float32)
    for j in range(tl.cdiv(N, BN)):
        kn = j * BN + n
        k = tl.load(K + base + kn[None, :] * D + d[:, None])
        v = tl.load(V + base + kn[:, None] * D + d[None, :])
        score = tl.dot(q, k) * SCALE
        m_new = tl.maximum(m, tl.max(score, 1))
        p = tl.exp(score - m_new[:, None])
        alpha = tl.exp(m - m_new)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l = l * alpha + tl.sum(p, 1)
        m = m_new
    output = acc / l[:, None]
    tl.store(O + base + qm[:, None] * D + d[None, :], output)
    tl.store(LSE + bh * N + qm, m + tl.log(l))


@triton.jit
def _flash_bwd_dq(Q, K, V, O, DO, LSE, DQ,
                  N: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr):
    qm = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    kn = tl.arange(0, BN)
    d = tl.arange(0, D)
    base = bh * N * D
    q = tl.load(Q + base + qm[:, None] * D + d[None, :])
    do = tl.load(DO + base + qm[:, None] * D + d[None, :])
    o = tl.load(O + base + qm[:, None] * D + d[None, :])
    lse = tl.load(LSE + bh * N + qm)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    dq = tl.full((BM, D), 0.0, tl.float32)
    for j in range(tl.cdiv(N, BN)):
        kj = j * BN + kn
        kt = tl.load(K + base + kj[None, :] * D + d[:, None])
        vt = tl.load(V + base + kj[None, :] * D + d[:, None])
        logits = tl.dot(q, kt) * SCALE
        p = tl.exp(logits - lse[:, None])
        dp = tl.dot(do, vt)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(q.dtype), tl.trans(kt))
    tl.store(DQ + base + qm[:, None] * D + d[None, :], dq * SCALE)


@triton.jit
def _flash_bwd_dkdv(Q, K, V, O, DO, LSE, DK, DV,
                    N: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr):
    kn = tl.program_id(0) * BN + tl.arange(0, BN)
    bh = tl.program_id(1)
    qm = tl.arange(0, BM)
    d = tl.arange(0, D)
    base = bh * N * D
    k = tl.load(K + base + kn[:, None] * D + d[None, :])
    v = tl.load(V + base + kn[:, None] * D + d[None, :])
    dk = tl.full((BN, D), 0.0, tl.float32)
    dv = tl.full((BN, D), 0.0, tl.float32)
    for j in range(tl.cdiv(N, BM)):
        qj = j * BM + qm
        q = tl.load(Q + base + qj[:, None] * D + d[None, :])
        do = tl.load(DO + base + qj[:, None] * D + d[None, :])
        o = tl.load(O + base + qj[:, None] * D + d[None, :])
        lse = tl.load(LSE + bh * N + qj)
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
        logits = tl.dot(q, tl.trans(k)) * SCALE
        p = tl.exp(logits - lse[:, None])
        dv += tl.dot(tl.trans(p.to(q.dtype)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    tl.store(DK + base + kn[:, None] * D + d[None, :], dk * SCALE)
    tl.store(DV + base + kn[:, None] * D + d[None, :], dv)


class _CustomFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        if not (q.device.type == 'cuda' and q.shape == k.shape == v.shape
                and q.ndim == 4 and q.is_contiguous() and k.is_contiguous()
                and v.is_contiguous() and q.dtype == k.dtype == v.dtype == torch.float16):
            raise ValueError('Custom flash requires contiguous CUDA FP16 [B,H,N,D] Q,K,V')
        b, h, n, d = q.shape
        if n % 64 or d not in (32, 64):
            raise ValueError('Custom flash requires sequence length divisible by 64, D=32/64')
        o = torch.empty_like(q)
        lse = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
        scale = d ** -0.5
        _flash_fwd[(triton.cdiv(n, 32), b * h)](q, k, v, o, lse, n, d, scale, 32, 64, num_warps=4)
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()
        b, h, n, d = q.shape
        scale = d ** -0.5
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_bwd_dq[(triton.cdiv(n, 32), b * h)](
            q, k, v, o, do, lse, dq, n, d, scale, 32, 64, num_warps=4)
        _flash_bwd_dkdv[(triton.cdiv(n, 64), b * h)](
            q, k, v, o, do, lse, dk, dv, n, d, scale, 32, 64, num_warps=4)
        return dq, dk, dv


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Training-capable custom FlashAttention, no SDPA fallback."""
    return _CustomFlashAttention.apply(q, k, v)


def check_correctness(device='cuda'):
    """Fails loudly rather than training with a faulty custom kernel."""
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for kernel correctness tests.')
    for n in (128, 256):
        for d in (32, 64):
            q, k, v = [torch.randn((2, 4, n, d), device=device, dtype=torch.float16,
                                   requires_grad=True) * 0.45 for _ in range(3)]
            q, k, v = [x.detach().requires_grad_(True) for x in (q, k, v)]
            out = flash_attention(q, k, v)
            grad = torch.randn_like(out)
            ours = torch.autograd.grad(out, (q, k, v), grad)
            qr, kr, vr = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
            ref = F.scaled_dot_product_attention(qr, kr, vr)
            theirs = torch.autograd.grad(ref, (qr, kr, vr), grad)
            max_abs = (out.float() - ref.float()).abs().max().item()
            if not torch.allclose(out.float(), ref.float(), rtol=0.03, atol=0.008):
                raise AssertionError(f'Forward incorrect: N={n} D={d} max_abs={max_abs:.4g}')
            for name, a, z in zip('qkv', ours, theirs):
                relative = ((a.float()-z.float()).norm() / z.float().norm().clamp_min(1e-8)).item()
                if relative > 0.035:
                    raise AssertionError(f'Gradient {name} incorrect N={n} D={d} relL2={relative:.4f}')
            print(f'PASS N={n} D={d}, forward max_abs={max_abs:.4g}; gradients within 3.5% rel L2')
    print('All custom FlashAttention forward/backward checks passed.')
