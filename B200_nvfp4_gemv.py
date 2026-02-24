import os

import torch
from torch.utils.cpp_extension import load_inline
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>
#include <cstdint>

template<int NUM_WARPS_, int TPR_, int STAGES_, int MTILE_, int KTILE_, bool L2_PREFETCH_ = false, int MIN_BLOCKS_ = 1>
struct Cfg {

    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int TPR = TPR_;
    static constexpr int STAGES = STAGES_;
    static constexpr int MTILE = MTILE_;
    static constexpr int KTILE = KTILE_;
    static constexpr bool L2_PREFETCH = L2_PREFETCH_;
    static constexpr int MIN_BLOCKS = MIN_BLOCKS_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int THREADS = NUM_WARPS * WARP_SIZE;
    static constexpr int SF_VEC = 16;
    static constexpr int SF_K = KTILE / SF_VEC;
    static constexpr int K_BYTES = KTILE / 2;

    static constexpr bool USE_U64 = (K_BYTES > 1024);
    static constexpr bool USE_U32 = (K_BYTES > 256) && !USE_U64;
    static constexpr int K_DIV = USE_U64 ? 8 : (USE_U32 ? 4 : 1);
    static constexpr int K_COORD = K_BYTES / K_DIV;

    static constexpr int ROWS_PER_BLK = THREADS / TPR;
    static constexpr int CHUNKS = K_BYTES / 16 / TPR;
    static constexpr int PASSES = (MTILE + ROWS_PER_BLK - 1) / ROWS_PER_BLK;

    static constexpr uint32_t TILE_A = MTILE * K_BYTES;
    static constexpr uint32_t TILE_B = K_BYTES;
    static constexpr uint32_t TILE_SFA = MTILE * SF_K;
    static constexpr uint32_t TILE_SFB = SF_K;
    static constexpr uint32_t TILE_MBAR = sizeof(unsigned long long);
    static constexpr uint32_t TX_TOTAL = TILE_A + TILE_B + TILE_SFA + SF_K;
};

template<typename C>
struct alignas(128) SMem {
    alignas(128) unsigned char a[C::STAGES][C::TILE_A];
    alignas(128) unsigned char b[C::STAGES][C::TILE_B];
    alignas(128) unsigned char sfa[C::STAGES][C::TILE_SFA];

    static constexpr int SFB_STRIDE = C::SF_K < 128 ? 128 : C::SF_K;
    alignas(128) unsigned char sfb[C::STAGES][SFB_STRIDE];
    alignas(16)  unsigned long long mbar[C::STAGES];
};

#define CHECK_DRV(x) do { CUresult r = (x); TORCH_CHECK(r == CUDA_SUCCESS, "DRV err:", (int)r); } while(0)

static PFN_cuTensorMapEncodeTiled_v12000 get_tma_fn() {
    static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
    if (!fn) {
        cudaDriverEntryPointQueryResult st;
        void* p = nullptr;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &st);
        fn = (PFN_cuTensorMapEncodeTiled_v12000)p;
    }
    return fn;
}

__device__ __forceinline__ float fp4_dot(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint16_t sfa, uint16_t sfb)
{
    uint32_t r;
    asm volatile(
        "{\n"
        ".reg .b8 xa<8>, xb<8>;\n"
        ".reg .f16x2 ha<4>, hb<4>, c<4>, sf, sfa_v, sfb_v, m0, m1;\n"
        ".reg .f16 s0, s1, t0, t1, res;\n"
        ".reg .f32 rf;\n"

        "cvt.rn.f16x2.e4m3x2 sfa_v, %1;\n"
        "cvt.rn.f16x2.e4m3x2 sfb_v, %2;\n"
        "mul.rn.f16x2 sf, sfa_v, sfb_v;\n"
        "mov.b32 {s0,s1}, sf;\n"
        "mov.b32 m0, {s0,s0};\n"
        "mov.b32 m1, {s1,s1};\n"

        "mov.b32 c0, 0; mov.b32 c1, 0; mov.b32 c2, 0; mov.b32 c3, 0;\n"

        "mov.b32 {xa0,xa1,xa2,xa3}, %3; mov.b32 {xb0,xb1,xb2,xb3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 ha0, xa0; cvt.rn.f16x2.e2m1x2 ha1, xa1;\n"
        "cvt.rn.f16x2.e2m1x2 ha2, xa2; cvt.rn.f16x2.e2m1x2 ha3, xa3;\n"
        "cvt.rn.f16x2.e2m1x2 hb0, xb0; cvt.rn.f16x2.e2m1x2 hb1, xb1;\n"
        "cvt.rn.f16x2.e2m1x2 hb2, xb2; cvt.rn.f16x2.e2m1x2 hb3, xb3;\n"
        "fma.rn.f16x2 c0, ha0, hb0, c0; fma.rn.f16x2 c0, ha1, hb1, c0;\n"
        "fma.rn.f16x2 c0, ha2, hb2, c0; fma.rn.f16x2 c0, ha3, hb3, c0;\n"

        "mov.b32 {xa4,xa5,xa6,xa7}, %5; mov.b32 {xb4,xb5,xb6,xb7}, %6;\n"
        "cvt.rn.f16x2.e2m1x2 ha0, xa4; cvt.rn.f16x2.e2m1x2 ha1, xa5;\n"
        "cvt.rn.f16x2.e2m1x2 ha2, xa6; cvt.rn.f16x2.e2m1x2 ha3, xa7;\n"
        "cvt.rn.f16x2.e2m1x2 hb0, xb4; cvt.rn.f16x2.e2m1x2 hb1, xb5;\n"
        "cvt.rn.f16x2.e2m1x2 hb2, xb6; cvt.rn.f16x2.e2m1x2 hb3, xb7;\n"
        "fma.rn.f16x2 c1, ha0, hb0, c1; fma.rn.f16x2 c1, ha1, hb1, c1;\n"
        "fma.rn.f16x2 c1, ha2, hb2, c1; fma.rn.f16x2 c1, ha3, hb3, c1;\n"

        "mov.b32 {xa0,xa1,xa2,xa3}, %7; mov.b32 {xb0,xb1,xb2,xb3}, %8;\n"
        "cvt.rn.f16x2.e2m1x2 ha0, xa0; cvt.rn.f16x2.e2m1x2 ha1, xa1;\n"
        "cvt.rn.f16x2.e2m1x2 ha2, xa2; cvt.rn.f16x2.e2m1x2 ha3, xa3;\n"
        "cvt.rn.f16x2.e2m1x2 hb0, xb0; cvt.rn.f16x2.e2m1x2 hb1, xb1;\n"
        "cvt.rn.f16x2.e2m1x2 hb2, xb2; cvt.rn.f16x2.e2m1x2 hb3, xb3;\n"
        "fma.rn.f16x2 c2, ha0, hb0, c2; fma.rn.f16x2 c2, ha1, hb1, c2;\n"
        "fma.rn.f16x2 c2, ha2, hb2, c2; fma.rn.f16x2 c2, ha3, hb3, c2;\n"

        "mov.b32 {xa4,xa5,xa6,xa7}, %9; mov.b32 {xb4,xb5,xb6,xb7}, %10;\n"
        "cvt.rn.f16x2.e2m1x2 ha0, xa4; cvt.rn.f16x2.e2m1x2 ha1, xa5;\n"
        "cvt.rn.f16x2.e2m1x2 ha2, xa6; cvt.rn.f16x2.e2m1x2 ha3, xa7;\n"
        "cvt.rn.f16x2.e2m1x2 hb0, xb4; cvt.rn.f16x2.e2m1x2 hb1, xb5;\n"
        "cvt.rn.f16x2.e2m1x2 hb2, xb6; cvt.rn.f16x2.e2m1x2 hb3, xb7;\n"
        "fma.rn.f16x2 c3, ha0, hb0, c3; fma.rn.f16x2 c3, ha1, hb1, c3;\n"
        "fma.rn.f16x2 c3, ha2, hb2, c3; fma.rn.f16x2 c3, ha3, hb3, c3;\n"

        "add.rn.f16x2 c0, c0, c1; add.rn.f16x2 c2, c2, c3;\n"
        "mul.rn.f16x2 c0, m0, c0; mul.rn.f16x2 c2, m1, c2;\n"
        "add.rn.f16x2 c0, c0, c2;\n"
        "mov.b32 {t0,t1}, c0;\n"
        "add.rn.f16 res, t0, t1;\n"
        "cvt.f32.f16 rf, res;\n"
        "mov.b32 %0, rf;\n"
        "}\n"
        : "=r"(r) : "h"(sfa), "h"(sfb),
          "r"(a0), "r"(b0), "r"(a1), "r"(b1),
          "r"(a2), "r"(b2), "r"(a3), "r"(b3));
    return __int_as_float(r);
}

__device__ __forceinline__ void mbar_init(uint32_t a, int c) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"::"r"(a),"r"(c));
}
__device__ __forceinline__ void mbar_tx(uint32_t a, uint32_t b) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"::"r"(a),"r"(b));
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int p) {
    asm volatile("{\n.reg .pred q;\nL_%=: mbarrier.try_wait.parity.shared::cta.b64 q,[%0],%1; @!q bra L_%=;\n}\n"::"r"(a),"r"(p));
}

__device__ __forceinline__ void tma_3d(uint32_t d, uint64_t t, uint32_t m, int x, int y, int z) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4,%5}],[%2];\n"
        ::"r"(d),"l"(t),"r"(m),"r"(x),"r"(y),"r"(z):"memory");
}
__device__ __forceinline__ void tma_2d(uint32_t d, uint64_t t, uint32_t m, int x, int y) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];\n"
        ::"r"(d),"l"(t),"r"(m),"r"(x),"r"(y):"memory");
}
__device__ __forceinline__ void tma_3d_L2(uint32_t d, uint64_t t, uint32_t m, int x, int y, int z, uint64_t h) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%3,%4,%5}],[%2],%6;\n"
        ::"r"(d),"l"(t),"r"(m),"r"(x),"r"(y),"r"(z),"l"(h):"memory");
}
__device__ __forceinline__ void tma_2d_L2(uint32_t d, uint64_t t, uint32_t m, int x, int y, uint64_t h) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%3,%4}],[%2],%5;\n"
        ::"r"(d),"l"(t),"r"(m),"r"(x),"r"(y),"l"(h):"memory");
}
__device__ __forceinline__ void pf_3d(uint64_t t, int x, int y, int z, uint64_t h) {
    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0,{%1,%2,%3}],%4;\n"
        ::"l"(t),"r"(x),"r"(y),"r"(z),"l"(h):"memory");
}
__device__ __forceinline__ void pf_2d(uint64_t t, int x, int y, uint64_t h) {
    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint [%0,{%1,%2}],%3;\n"
        ::"l"(t),"r"(x),"r"(y),"l"(h):"memory");
}

template<typename C>
void enc_a(CUtensorMap* d, const at::Tensor& t, int64_t M, int64_t K, int64_t L) {
    auto s = t.strides();
    CUtensorMapDataType dtype;
    cuuint64_t gd0;
    cuuint32_t bd0;

    if constexpr (C::USE_U64) {
        dtype = CU_TENSOR_MAP_DATA_TYPE_UINT64;
        gd0 = K / 16;
        bd0 = C::K_COORD;
    } else if constexpr (C::USE_U32) {
        dtype = CU_TENSOR_MAP_DATA_TYPE_UINT32;
        gd0 = K / 8;
        bd0 = C::K_COORD;
    } else {
        dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        gd0 = K / 2;
        bd0 = C::K_COORD;
    }

    cuuint64_t gd[3] = {gd0, (cuuint64_t)M, (cuuint64_t)L};
    cuuint64_t gs[2] = {(cuuint64_t)s[0], (cuuint64_t)s[2]};
    cuuint32_t bd[3] = {bd0, (cuuint32_t)C::MTILE, 1}, es[3] = {1, 1, 1};

    CHECK_DRV(get_tma_fn()(d, dtype, 3, t.data_ptr(), gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

template<typename C>
void enc_b(CUtensorMap* d, const at::Tensor& t, int64_t K, int64_t L) {
    auto s = t.strides();
    CUtensorMapDataType dtype;
    cuuint64_t gd0;
    cuuint32_t bd0;

    if constexpr (C::USE_U64) {
        dtype = CU_TENSOR_MAP_DATA_TYPE_UINT64;
        gd0 = K / 16;
        bd0 = C::K_COORD;
    } else if constexpr (C::USE_U32) {
        dtype = CU_TENSOR_MAP_DATA_TYPE_UINT32;
        gd0 = K / 8;
        bd0 = C::K_COORD;
    } else {
        dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        gd0 = K / 2;
        bd0 = C::K_COORD;
    }

    cuuint64_t gd[2] = {gd0, (cuuint64_t)L};
    cuuint64_t gs[1] = {(cuuint64_t)s[2]};
    cuuint32_t bd[2] = {bd0, 1}, es[2] = {1, 1};

    CHECK_DRV(get_tma_fn()(d, dtype, 2, t.data_ptr(), gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

template<typename C>
void enc_sfa(CUtensorMap* d, const at::Tensor& t, int64_t M, int64_t Ks, int64_t L) {
    auto s = t.strides();
    cuuint64_t gd[3] = {(cuuint64_t)Ks, (cuuint64_t)M, (cuuint64_t)L};
    cuuint64_t gs[2] = {(cuuint64_t)s[0], (cuuint64_t)s[2]};
    cuuint32_t bd[3] = {(cuuint32_t)C::SF_K, (cuuint32_t)C::MTILE, 1}, es[3] = {1, 1, 1};
    CHECK_DRV(get_tma_fn()(d, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, t.data_ptr(), gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

template<typename C>
void enc_sfb(CUtensorMap* d, const at::Tensor& t, int64_t Ks, int64_t L) {
    auto s = t.strides();
    cuuint64_t gd[2] = {(cuuint64_t)Ks, (cuuint64_t)L};
    cuuint64_t gs[1] = {(cuuint64_t)s[2]};
    cuuint32_t bd[2] = {(cuuint32_t)C::SF_K, 1}, es[2] = {1, 1};
    CHECK_DRV(get_tma_fn()(d, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(), gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

template<typename C>
__global__ void __launch_bounds__(C::THREADS, C::MIN_BLOCKS)
kernel(const __grid_constant__ CUtensorMap ta, const __grid_constant__ CUtensorMap tb,
       const __grid_constant__ CUtensorMap tsa, const __grid_constant__ CUtensorMap tsb,
       __half* __restrict__ out, int64_t M, int64_t K, int64_t L, int64_t sm, int64_t sl)
{
#if __CUDA_ARCH__ < 900
    return;
#endif
    const int li = blockIdx.y;
    if (li >= L) return;

    const int nkt = (int)(K/2) / C::K_BYTES;
    const int Mt  = (int)(M / C::MTILE);
    if (!nkt || !Mt) return;

    extern __shared__ __align__(128) char mem[];
    auto* sh = (SMem<C>*)mem;

    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    const int row_w= lane / C::TPR;
    const int lin  = lane % C::TPR;
    const int rpw  = 32 / C::TPR;
    const int br   = wid * rpw + row_w;

    const uint64_t da  = (uint64_t)&ta;
    const uint64_t db  = (uint64_t)&tb;
    const uint64_t dsa = (uint64_t)&tsa;
    const uint64_t dsb = (uint64_t)&tsb;

    uint32_t aa[C::STAGES], ab[C::STAGES], asa[C::STAGES], asb[C::STAGES], am[C::STAGES];
    #pragma unroll
    for (int s = 0; s < C::STAGES; ++s) {
        aa[s]  = __cvta_generic_to_shared(sh->a[s]);
        ab[s]  = __cvta_generic_to_shared(sh->b[s]);
        asa[s] = __cvta_generic_to_shared(sh->sfa[s]);
        asb[s] = __cvta_generic_to_shared(sh->sfb[s]);
        am[s]  = __cvta_generic_to_shared(&sh->mbar[s]);
    }

    constexpr uint64_t H1 = 1;
    constexpr uint64_t H2 = 2;

    int ph[C::STAGES];
    if constexpr (C::STAGES >= 2) {
        if (tid == 0) {
            #pragma unroll
            for (int s = 0; s < C::STAGES; ++s) {
                mbar_init(am[s], 1);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int s = 0; s < C::STAGES; ++s) {
            ph[s] = 0;
        }
    }

    for (int tm = blockIdx.x; tm < Mt; tm += gridDim.x) {
        const int mb = tm * C::MTILE;

        float acc[C::PASSES];
        #pragma unroll
        for (int p = 0; p < C::PASSES; ++p) {
            acc[p] = 0.f;
        }

        if constexpr (C::L2_PREFETCH) {
            if (tid == 0) {
                #pragma unroll
                for (int pf = 0; pf < 2 && pf < nkt; ++pf) {
                    pf_3d(da,  pf * C::K_COORD, mb, li, H1);
                    pf_2d(db,  pf * C::K_COORD, li,     H2);
                    pf_3d(dsa, pf * C::SF_K,    mb, li, H1);
                    pf_2d(dsb, pf * C::SF_K,    li,     H2);
                }
            }
        }

        if constexpr (C::STAGES == 1) {
            for (int kt = 0; kt < nkt; ++kt) {
                __syncthreads();

                if (tid == 0) {
                    mbar_init(am[0], 1);
                }__syncthreads();
                if (tid == 0) {
                    mbar_tx(am[0], C::TX_TOTAL);

                    int kc = kt * C::K_COORD;
                    int ks = kt * C::SF_K;

                    if constexpr (C::L2_PREFETCH) {
                        int fk = kt + 1;
                        if (fk < nkt) {
                            pf_3d(da,  fk * C::K_COORD, mb, li, H1);
                            pf_2d(db,  fk * C::K_COORD, li,     H2);
                            pf_3d(dsa, fk * C::SF_K,    mb, li, H1);
                            pf_2d(dsb, fk * C::SF_K,    li,     H2);
                        }

                        tma_3d_L2(aa[0],  da,  am[0], kc, mb, li, H1);
                        tma_2d_L2(ab[0],  db,  am[0], kc, li,     H2);
                        tma_3d_L2(asa[0], dsa, am[0], ks, mb, li, H1);
                        tma_2d_L2(asb[0], dsb, am[0], ks, li,     H2);
                    } else {
                        tma_3d(aa[0],  da,  am[0], kc, mb, li);
                        tma_2d(ab[0],  db,  am[0], kc, li);
                        tma_3d(asa[0], dsa, am[0], ks, mb, li);
                        tma_2d(asb[0], dsb, am[0], ks, li);
                    }
                }

                mbar_wait(am[0], 0);
                __syncthreads();

                const unsigned char* __restrict__ pA  = sh->a[0];
                const unsigned char* __restrict__ pB  = sh->b[0];
                const unsigned char* __restrict__ pSA = sh->sfa[0];
                const unsigned char* __restrict__ pSB = sh->sfb[0];

                #pragma unroll
                for (int c = 0; c < C::CHUNKS; ++c) {
                    const int ci = c * C::TPR + lin;
                    const int ko = ci * 16;
                    const int si = ci * 2;

                    const uint4     fB = *(const uint4*)(&pB[ko]);
                    const uint16_t  sb = *(const uint16_t*)(&pSB[si]);

                    #pragma unroll
                    for (int p = 0; p < C::PASSES; ++p) {
                        const int row = br + p * C::ROWS_PER_BLK;
                        if (row < C::MTILE) {
                            const uint4     fA = *(const uint4*)(&pA[row * C::K_BYTES + ko]);
                            const uint16_t  sa = *(const uint16_t*)(&pSA[row * C::SF_K + si]);

                            acc[p] += fp4_dot(fA.x, fA.y, fA.z, fA.w,
                                              fB.x, fB.y, fB.z, fB.w,
                                              sa, sb);
                        }
                    }
                }

                __syncthreads();
            }
        }

        else {
            const int pro = min(nkt, C::STAGES - 1);
            for (int s = 0; s < pro; ++s) {
                if (tid == 0) {
                    mbar_tx(am[s], C::TX_TOTAL);
                    int kc = s * C::K_COORD;
                    int ks = s * C::SF_K;
                    if constexpr (C::L2_PREFETCH) {
                        tma_3d_L2(aa[s],  da,  am[s], kc, mb, li, H1);
                        tma_2d_L2(ab[s],  db,  am[s], kc, li,     H2);
                        tma_3d_L2(asa[s], dsa, am[s], ks, mb, li, H1);
                        tma_2d_L2(asb[s], dsb, am[s], ks, li,     H2);
                    } else {
                        tma_3d(aa[s],  da,  am[s], kc, mb, li);
                        tma_2d(ab[s],  db,  am[s], kc, li);
                        tma_3d(asa[s], dsa, am[s], ks, mb, li);
                        tma_2d(asb[s], dsb, am[s], ks, li);
                    }
                }
            }

            if (pro > 0) {
                mbar_wait(am[0], ph[0]);
                ph[0] ^= 1;
            }
            __syncthreads();

            for (int kt = 0; kt < nkt; ++kt) {
                const int cs = kt % C::STAGES;
                const int nk = kt + C::STAGES - 1;
                const int ns = nk % C::STAGES;

                if (nk < nkt && tid == 0) {
                    mbar_tx(am[ns], C::TX_TOTAL);
                    int kc = nk * C::K_COORD;
                    int ks = nk * C::SF_K;
                    if constexpr (C::L2_PREFETCH) {
                        tma_3d_L2(aa[ns],  da,  am[ns], kc, mb, li, H1);
                        tma_2d_L2(ab[ns],  db,  am[ns], kc, li,     H2);
                        tma_3d_L2(asa[ns], dsa, am[ns], ks, mb, li, H1);
                        tma_2d_L2(asb[ns], dsb, am[ns], ks, li,     H2);
                        int fk = nk + 2;
                        if (fk < nkt) {
                            pf_3d(da,  fk * C::K_COORD, mb, li, H1);
                            pf_2d(db,  fk * C::K_COORD, li,     H2);
                            pf_3d(dsa, fk * C::SF_K,    mb, li, H1);
                            pf_2d(dsb, fk * C::SF_K,    li,     H2);
                        }
                    } else {
                        tma_3d(aa[ns],  da,  am[ns], kc, mb, li);
                        tma_2d(ab[ns],  db,  am[ns], kc, li);
                        tma_3d(asa[ns], dsa, am[ns], ks, mb, li);
                        tma_2d(asb[ns], dsb, am[ns], ks, li);
                    }
                }

                const unsigned char* __restrict__ pA  = sh->a[cs];
                const unsigned char* __restrict__ pB  = sh->b[cs];
                const unsigned char* __restrict__ pSA = sh->sfa[cs];
                const unsigned char* __restrict__ pSB = sh->sfb[cs];

                #pragma unroll
                for (int c = 0; c < C::CHUNKS; ++c) {
                    const int ci = c * C::TPR + lin;
                    const int ko = ci * 16;
                    const int si = ci * 2;

                    const uint4     fB = *(const uint4*)(&pB[ko]);
                    const uint16_t  sb = *(const uint16_t*)(&pSB[si]);

                    #pragma unroll
                    for (int p = 0; p < C::PASSES; ++p) {
                        const int row = br + p * C::ROWS_PER_BLK;
                        if (row < C::MTILE) {
                            const uint4     fA = *(const uint4*)(&pA[row * C::K_BYTES + ko]);
                            const uint16_t  sa = *(const uint16_t*)(&pSA[row * C::SF_K + si]);

                            acc[p] += fp4_dot(fA.x, fA.y, fA.z, fA.w,
                                              fB.x, fB.y, fB.z, fB.w,
                                              sa, sb);
                        }
                    }
                }

                if (kt + 1 < nkt) {
                    int ws = (kt + 1) % C::STAGES;
                    mbar_wait(am[ws], ph[ws]);
                    ph[ws] ^= 1;
                    __syncthreads();
                }
            }
        }

        #pragma unroll
        for (int p = 0; p < C::PASSES; ++p) {
            int row = br + p * C::ROWS_PER_BLK;
            if (row >= C::MTILE) continue;

            float v = acc[p];
            #pragma unroll
            for (int d = C::TPR >> 1; d > 0; d >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, d);
            }

            if (lin == 0) {
                int mo = mb + row;
                if (mo < M) {
                    out[mo * sm + li * sl] = __float2half(v);
                }
            }
        }
        __syncthreads();
    }
}

template<typename C>
void launch(const at::Tensor& a, const at::Tensor& b, const at::Tensor& sa, const at::Tensor& sb,
            at::Tensor& c, int64_t M, int64_t K, int64_t L) {
    CUtensorMap ta{}, tb{}, tsa{}, tsb{};
    enc_a<C>(&ta, a, M, K, L);
    enc_b<C>(&tb, b, K, L);
    enc_sfa<C>(&tsa, sa, M, K / C::SF_VEC, L);
    enc_sfb<C>(&tsb, sb, K / C::SF_VEC, L);

    int Mt = M / C::MTILE;

    int gx = Mt;

    dim3 grid(gx, (int)L);
    dim3 block(C::THREADS);
    size_t smem = sizeof(SMem<C>);

    auto kern = &kernel<C>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    auto cs = c.strides();
    kern<<<grid, block, smem>>>(
        ta, tb, tsa, tsb, (__half*)c.data_ptr(), M, K, L, cs[0], cs[2]);
}

void nvfp4_gemv_forward(const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& sa, const at::Tensor& sb, at::Tensor& c,
    int64_t M, int64_t K, int64_t L)
{
    if (M == 7168 && K == 16384 && L == 1) {

        using C = Cfg<8, 16, 1, 8, 4096, false, 7>;
        launch<C>(a, b, sa, sb, c, M, K, L);
    }
    else if (M == 7168 && K == 2048 && L == 4) {

        using C = Cfg<2, 8, 1, 8, 2048, false, 25>;
        launch<C>(a, b, sa, sb, c, M, K, L);
    }
    else if (M == 4096 && K == 7168 && L == 8) {

        using C = Cfg<2, 8, 1, 8, 1024, false, 28>;
        launch<C>(a, b, sa, sb, c, M, K, L);
    }
    else {
        using C = Cfg<4, 8, 1, 32, 256, false, 1>;
        launch<C>(a, b, sa, sb, c, M, K, L);
    }
}


"""

cpp_decl = r"""
#include <torch/extension.h>

void nvfp4_gemv_forward(const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& sa, const at::Tensor& sb, at::Tensor& c,
    int64_t M, int64_t K, int64_t L);
"""

gemv_module = load_inline(
    name='nvfp4_gemv_semi_optimized',
    cpp_sources=cpp_decl,
    cuda_sources=cuda_source,
    extra_cflags=['-O3'],
    extra_cuda_cflags=[
        '-std=c++17',
        '-O3',
        '-gencode=arch=compute_100a,code=sm_100a',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-DCUDA_PTX_FP4FP6_CVT_ENABLED=1',
        '--use_fast_math',
    ],
    extra_ldflags=['-lcuda'],
    with_cuda=True,
    functions=['nvfp4_gemv_forward'],
    verbose=False,
)

from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    a, b, sfa, sfb, _, _, c = data

    m, k, l = a.shape
    b_vec = b[0:1, :, :]
    sfb_vec = sfb[0:1, :, :]
    k_fp4 = k * 2

    gemv_module.nvfp4_gemv_forward(
        a,
        b_vec,
        sfa,
        sfb_vec,
        c,
        m,
        k_fp4,
        l
    )

    return c
