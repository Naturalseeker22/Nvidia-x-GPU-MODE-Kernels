# Nvidia x GPU-MODE Kernels

NVFP4 CUDA kernels for the Blackwell B200 GPU, written for the [Blackwell NVFP4 Kernel competition](https://luma.com/9n27uem4)

Three kernels — Grouped GEMM, Dual GEMM (with fused SiLU), and GEMV — all operating in FP4 (E2M1) with block-wise scaling on sm_100a.

For a deeper dive into the techniques used, check out the blog: [Nvidia x GPU-MODE Kernels](https://naturalseeker22.github.io/#/blog/gpu-mode-kernels-top-10)

---

## Kernels

### Grouped GEMM — `B200_nvfp4_Groupgemm.py`

Runs multiple matrix multiplications in one kernel launch. Each group can have its own M, N, K — up to 8 groups at once. Problem shapes are passed in dynamically, so it handles whatever you throw at it.

Optimized for these dispatch strategies:

| Groups | K       | Kernel variant              |
|--------|---------|-----------------------------|
| 1–4    | ≤ 2048  | CTA-based                   |
| 1–4    | > 2048  | Clustered (2-CTA)           |
| 5–8    | ≤ 2048  | Persistent                  |
| 5–8    | > 2048  | Persistent + Clustered      |

---

### Dual GEMM + SiLU — `B200_nvfp4_dual_gemm.py`

Two GEMMs sharing the same A matrix, with SiLU fused in the epilogue — the typical MLP gate pattern in transformers:

```
output = SiLU(A @ B1) * (A @ B2)
```

Optimized for these K dimensions: **7168, 4096, 2304, 2048, 1536, 512, 256**

---

### GEMV — `B200_nvfp4_gemv.py`

Matrix-vector multiply for small batch sizes (L = 1–8), the kind you hit during token-by-token LLM inference:

Optimized for these shapes:

| M | K | L |
|------|-------|---|
| 7168 | 16384 | 1 |
| 7168 | 2048  | 4 |
| 4096 | 7168  | 8 |

Falls back to a general config for anything else.

---

## Requirements

- **GPU**: Blackwell (B200 / B100 / GB200) — sm_100a
- **CUDA**: 12.8+
- **PyTorch**: 2.x with CUDA
- **Python**: 3.8+

## Running

Each file is self-contained and JIT-compiles via `torch.utils.cpp_extension.load_inline`:

```bash
export TORCH_CUDA_ARCH_LIST="10.0a"

python B200_nvfp4_Groupgemm.py
python B200_nvfp4_dual_gemm.py
python B200_nvfp4_gemv.py
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
