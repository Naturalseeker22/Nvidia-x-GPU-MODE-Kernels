import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = r'''
// Target: NVIDIA Blackwell (sm_100a)
// Optimized dual GEMM kernel with SiLU fusion

#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>

template<int BlockM, int BlockN, int BlockK, int Stages>
struct KernelConfig {
    static constexpr int WARP_SIZE = 32;
    static constexpr int MMA_K     = 64;

    static constexpr int BM = BlockM;
    static constexpr int BN = BlockN;
    static constexpr int BK = BlockK;
    static constexpr int NUM_STAGES = Stages;

    static constexpr int A_BYTES     = BM * BK / 2;
    static constexpr int B_BYTES     = BN * BK / 2;
    static constexpr int SF_BYTES    = 128 * BK / 16;
    static constexpr int STAGE_BYTES = A_BYTES + 2 * B_BYTES + 3 * SF_BYTES;

    static constexpr int NUM_WARPS = BM / WARP_SIZE + 2;
    static constexpr int THREADS   = BM + 2 * WARP_SIZE;

    static constexpr int SF_COLS   = 4 * (BK / MMA_K);
    static constexpr int TMEM_SFA  = 2 * BN;
    static constexpr int TMEM_SFB1 = TMEM_SFA + SF_COLS;
    static constexpr int TMEM_SFB2 = TMEM_SFB1 + SF_COLS;
    static constexpr int TMEM_COLS = 512;

    static constexpr uint32_t MMA_IDESC = (1U << 7) | (1U << 10) | 
                                          ((uint32_t)BN >> 3 << 17) | (1U << 27);
};

namespace math {

__device__ __forceinline__ float fast_sigmoid(float x) {
    float e, r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(-1.442695041f * x));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(e + 1.0f));
    return r;
}

__device__ __forceinline__ float silu(float x) {
    return x * fast_sigmoid(x);
}

} // namespace math

namespace desc {

__device__ __forceinline__ uint64_t encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t matrix_header() {
    return (encode(1024) << 32) | (1ULL << 46) | (2ULL << 61);
}

__device__ __forceinline__ uint64_t matrix(int smem_addr) {
    return encode(smem_addr) | matrix_header();
}

__device__ __forceinline__ uint64_t scale_header() {
    return (encode(128) << 32) | (1ULL << 46);
}

__device__ __forceinline__ uint64_t scale(int smem_addr) {
    return encode(smem_addr) | scale_header();
}

} // namespace desc

namespace barrier {

__device__ __forceinline__ uint32_t elect_one() {
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .pred px;\n"
        "elect.sync _|px, 0xFFFFFFFF;\n"
        "@px mov.s32 %0, 1;\n"
        "}" : "+r"(pred));
    return pred;
}

__device__ __forceinline__ void bar_init(int addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void bar_wait(int addr, int phase) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, 0x989680;\n"
        "@p bra DONE;\n"
        "bra WAIT;\n"
        "DONE:\n"
        "}" :: "r"(addr), "r"(phase));
}

__device__ __forceinline__ void bar_arrive_tx(int addr, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(bytes) : "memory");
}

} // namespace barrier

namespace tma {

// L2 Cache Hints - use strategic hints based on data access patterns
constexpr uint64_t L2_NORMAL     = 0x1000000000000000ULL;  // Normal eviction
constexpr uint64_t L2_TREAMING  = 0x12F0000000000000ULL;  // evict_first - ming data
constexpr uint64_t L2_PERSISTENT = 0x14F0000000000000ULL;  // evict_last - frequently reused

__device__ __forceinline__ void prefetch_desc(const void* desc) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(desc) : "memory");
}

__device__ __forceinline__ void prefetch_3d(const void* desc, int x, int y, int z) {
    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global [%0, {%1, %2, %3}];"
                 :: "l"(desc), "r"(x), "r"(y), "r"(z) : "memory");
}

__device__ __forceinline__ void prefetch_linear(const void* src, int bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                 :: "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void load_3d(int dst, const void* desc, int x, int y, int z, 
                                          int mbar, uint64_t hint = L2_NORMAL) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %6;"
        :: "r"(dst), "l"(desc), "r"(x), "r"(y), "r"(z), "r"(mbar), "l"(hint) : "memory");
}

__device__ __forceinline__ void load_linear(int dst, const void* src, int bytes, 
                                              int mbar, uint64_t hint = L2_NORMAL) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;"
        :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar), "l"(hint));
}

void encode_desc(CUtensorMap* desc, const char* ptr, 
                 uint64_t height, uint64_t width,
                 uint32_t tile_height, uint32_t tile_width) {
    uint64_t dims[3]    = {256, height, width / 256};
    uint64_t strides[2] = {width / 2, 128};
    uint32_t box[3]     = {256, tile_height, tile_width / 256};
    uint32_t elem[3]    = {1, 1, 1};
    
    cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3, 
                           (void*)ptr, dims, strides, box, elem,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, 
                           CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, 
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

} // namespace tma

namespace tmem {

__device__ __forceinline__ void alloc(int smem_addr, int cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr), "r"(cols));
}

__device__ __forceinline__ void dealloc(int base, int cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" 
                 :: "r"(base), "r"(cols));
}

__device__ __forceinline__ void copy_scale(int taddr, uint64_t sdesc) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" 
                 :: "r"(taddr), "l"(sdesc));
}

__device__ __forceinline__ void commit(int mbar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(mbar) : "memory");
}

__device__ __forceinline__ void fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ void wait_load() {
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

template<int N>
__device__ __forceinline__ void load(float* dst, int row, int col);

template<>
__device__ __forceinline__ void load<64>(float* t, int row, int col) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=f"(t[0]), "=f"(t[1]), "=f"(t[2]), "=f"(t[3]), "=f"(t[4]), "=f"(t[5]), "=f"(t[6]), "=f"(t[7]),
          "=f"(t[8]), "=f"(t[9]), "=f"(t[10]),"=f"(t[11]),"=f"(t[12]),"=f"(t[13]),"=f"(t[14]),"=f"(t[15]),
          "=f"(t[16]),"=f"(t[17]),"=f"(t[18]),"=f"(t[19]),"=f"(t[20]),"=f"(t[21]),"=f"(t[22]),"=f"(t[23]),
          "=f"(t[24]),"=f"(t[25]),"=f"(t[26]),"=f"(t[27]),"=f"(t[28]),"=f"(t[29]),"=f"(t[30]),"=f"(t[31])
        : "r"((row << 16) | col));
}

template<>
__device__ __forceinline__ void load<128>(float* t, int row, int col) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
        : "=f"(t[0]), "=f"(t[1]), "=f"(t[2]), "=f"(t[3]), "=f"(t[4]), "=f"(t[5]), "=f"(t[6]), "=f"(t[7]),
          "=f"(t[8]), "=f"(t[9]), "=f"(t[10]),"=f"(t[11]),"=f"(t[12]),"=f"(t[13]),"=f"(t[14]),"=f"(t[15]),
          "=f"(t[16]),"=f"(t[17]),"=f"(t[18]),"=f"(t[19]),"=f"(t[20]),"=f"(t[21]),"=f"(t[22]),"=f"(t[23]),
          "=f"(t[24]),"=f"(t[25]),"=f"(t[26]),"=f"(t[27]),"=f"(t[28]),"=f"(t[29]),"=f"(t[30]),"=f"(t[31]),
          "=f"(t[32]),"=f"(t[33]),"=f"(t[34]),"=f"(t[35]),"=f"(t[36]),"=f"(t[37]),"=f"(t[38]),"=f"(t[39]),
          "=f"(t[40]),"=f"(t[41]),"=f"(t[42]),"=f"(t[43]),"=f"(t[44]),"=f"(t[45]),"=f"(t[46]),"=f"(t[47]),
          "=f"(t[48]),"=f"(t[49]),"=f"(t[50]),"=f"(t[51]),"=f"(t[52]),"=f"(t[53]),"=f"(t[54]),"=f"(t[55]),
          "=f"(t[56]),"=f"(t[57]),"=f"(t[58]),"=f"(t[59]),"=f"(t[60]),"=f"(t[61]),"=f"(t[62]),"=f"(t[63])
        : "r"((row << 16) | col));
}

} // namespace tmem

namespace mma {

__device__ __forceinline__ void nvfp4_fill(uint64_t a, uint64_t b, uint32_t idesc,
                                             int d, int sfa, int sfb, int acc) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill "
        "[%0], %1, %2, %3, [%4], [%5], p;\n"
        "}" :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc));
}

__device__ __forceinline__ void nvfp4_lastuse(uint64_t a, uint64_t b, uint32_t idesc,
                                                int d, int sfa, int sfb, int acc) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse "
        "[%0], %1, %2, %3, [%4], [%5], p;\n"
        "}" :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc));
}

} // namespace mma

__device__ __forceinline__ void store_cs(half* addr, half2 v) {
    asm volatile("st.global.cs.b32 [%0], %1;" :: "l"(addr), "r"(*reinterpret_cast<uint32_t*>(&v)) : "memory");
}

template<int K, typename Cfg>
__global__ __launch_bounds__(Cfg::THREADS)
void dual_gemm_kernel(
    const __grid_constant__ CUtensorMap desc_a,
    const __grid_constant__ CUtensorMap desc_b1,
    const __grid_constant__ CUtensorMap desc_b2,
    const char* __restrict__ sf_a,
    const char* __restrict__ sf_b1,
    const char* __restrict__ sf_b2,
    half* __restrict__ out,
    int M, int N)
{
    constexpr int BM = Cfg::BM;
    constexpr int BN = Cfg::BN;
    constexpr int BK = Cfg::BK;
    constexpr int STAGES = Cfg::NUM_STAGES;
    constexpr int NUM_ITERS = K / BK;
    
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int lane = tid % Cfg::WARP_SIZE;
    const int warp = tid / Cfg::WARP_SIZE;

    const int grid_m = M / BM;
    const int tile_m = bid % grid_m;
    const int tile_n = bid / grid_m;
    const int off_m  = tile_m * BM;
    const int off_n  = tile_n * BN;

    extern __shared__ __align__(1024) char smem[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem));

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t barriers[STAGES * 2 + 1];
    const int mbar_tma  = static_cast<int>(__cvta_generic_to_shared(barriers));
    const int mbar_mma  = mbar_tma + STAGES * 8;
    const int mbar_done = mbar_mma + STAGES * 8;

    // Warp 0: init barriers + enhanced prefetching
    if (warp == 0 && barrier::elect_one()) {
        tma::prefetch_desc(&desc_a);
        tma::prefetch_desc(&desc_b1);
        tma::prefetch_desc(&desc_b2);

        // Prefetch first iteration
        tma::prefetch_3d(&desc_a,  0, off_m, 0);
        tma::prefetch_3d(&desc_b1, 0, off_n, 0);
        tma::prefetch_3d(&desc_b2, 0, off_n, 0);
        
        // Prefetch second iteration to hide more latency
        if constexpr (NUM_ITERS >= 2) {
            const int z1 = (BK == 256) ? 1 : BK / 256;
            tma::prefetch_3d(&desc_a,  0, off_m, z1);
            tma::prefetch_3d(&desc_b1, 0, off_n, z1);
            tma::prefetch_3d(&desc_b2, 0, off_n, z1);
        }

        const int sf_stride = K / 64;
        tma::prefetch_linear(sf_a  + ((off_m / 128) * sf_stride) * 512, Cfg::SF_BYTES);
        tma::prefetch_linear(sf_b1 + ((off_n / 128) * sf_stride) * 512, Cfg::SF_BYTES);
        tma::prefetch_linear(sf_b2 + ((off_n / 128) * sf_stride) * 512, Cfg::SF_BYTES);

        #pragma unroll
        for (int i = 0; i < STAGES * 2 + 1; i++)
            barrier::bar_init(mbar_tma + i * 8, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp == 1) {
        tmem::alloc(smem_base, Cfg::TMEM_COLS);
    }
    __syncthreads();

    // TMA producer warp with optimized L2 hints
    if (warp == Cfg::NUM_WARPS - 2 && barrier::elect_one()) {
        const int sf_stride = K / 64;
        const int sf_m_offset = (off_m / 128) * sf_stride;
        const int sf_n_offset = (off_n / 128) * sf_stride;
        
        const char* sfa_ptr  = sf_a  + sf_m_offset * 512;
        const char* sfb1_ptr = sf_b1 + sf_n_offset * 512;
        const char* sfb2_ptr = sf_b2 + sf_n_offset * 512;
        
        constexpr int SF_ITER_BYTES = (BK / 64) * 512;

        auto issue = [&](int iter, int stage) {
            const int mbar    = mbar_tma + stage * 8;
            const int s_a     = smem_base + stage * Cfg::STAGE_BYTES;
            const int s_b1    = s_a + Cfg::A_BYTES;
            const int s_b2    = s_b1 + Cfg::B_BYTES;
            const int s_sfa   = s_b2 + Cfg::B_BYTES;
            const int s_sfb1  = s_sfa + Cfg::SF_BYTES;
            const int s_sfb2  = s_sfb1 + Cfg::SF_BYTES;

            const int z_coord = (BK == 256) ? iter : (iter * BK) / 256;

            // Use treaming hint for matrices (accessed once per iteration)
            // Use persistent hint for scale factors (potential reuse across tiles)
            tma::load_3d(s_a,  &desc_a,  0, off_m, z_coord, mbar, tma::L2_TREAMING);
            tma::load_3d(s_b1, &desc_b1, 0, off_n, z_coord, mbar, tma::L2_TREAMING);
            tma::load_3d(s_b2, &desc_b2, 0, off_n, z_coord, mbar, tma::L2_TREAMING);

            tma::load_linear(s_sfa,  sfa_ptr  + iter * SF_ITER_BYTES, Cfg::SF_BYTES, mbar, tma::L2_PERSISTENT);
            tma::load_linear(s_sfb1, sfb1_ptr + iter * SF_ITER_BYTES, Cfg::SF_BYTES, mbar, tma::L2_PERSISTENT);
            tma::load_linear(s_sfb2, sfb2_ptr + iter * SF_ITER_BYTES, Cfg::SF_BYTES, mbar, tma::L2_PERSISTENT);

            barrier::bar_arrive_tx(mbar, Cfg::STAGE_BYTES);
        };

        // Issue initial stages
        #pragma unroll
        for (int i = 0; i < STAGES; i++)
            issue(i, i);

        // Steady state
        for (int i = STAGES; i < NUM_ITERS; i++) {
            const int stage = i % STAGES;
            barrier::bar_wait(mbar_mma + stage * 8, (i / STAGES - 1) % 2);
            issue(i, stage);
        }
    }

    // MMA consumer warp
    else if (warp == Cfg::NUM_WARPS - 1 && barrier::elect_one()) {
        const int tmem_sfa_base  = Cfg::TMEM_SFA  + (tile_m % (128 / BM)) * (BM / 32);
        const int tmem_sfb1_base = Cfg::TMEM_SFB1 + (tile_n % (128 / BN)) * (BN / 32);
        const int tmem_sfb2_base = Cfg::TMEM_SFB2 + (tile_n % (128 / BN)) * (BN / 32);

        const uint64_t m_desc_header = desc::matrix_header();
        const uint64_t s_desc_header = desc::scale_header();

        const int sa_offset   = 0;
        const int sb1_offset  = Cfg::A_BYTES;
        const int sb2_offset  = Cfg::A_BYTES + Cfg::B_BYTES;
        const int sfa_offset  = Cfg::A_BYTES + 2 * Cfg::B_BYTES;
        const int sfb1_offset = sfa_offset + Cfg::SF_BYTES;
        const int sfb2_offset = sfb1_offset + Cfg::SF_BYTES;

        const uint64_t desc_a_base  = m_desc_header | ((uint64_t)(smem_base + sa_offset) >> 4);
        const uint64_t desc_b1_base = m_desc_header | ((uint64_t)(smem_base + sb1_offset) >> 4);
        const uint64_t desc_b2_base = m_desc_header | ((uint64_t)(smem_base + sb2_offset) >> 4);

        const uint64_t desc_sfa_base  = s_desc_header | ((uint64_t)(smem_base + sfa_offset) >> 4);
        const uint64_t desc_sfb1_base = s_desc_header | ((uint64_t)(smem_base + sfb1_offset) >> 4);
        const uint64_t desc_sfb2_base = s_desc_header | ((uint64_t)(smem_base + sfb2_offset) >> 4);

        constexpr uint64_t STAGE_INC_DESC = Cfg::STAGE_BYTES >> 4;
        constexpr uint64_t STRIDE_A       = (uint64_t)(BM * 128) >> 4; 
        constexpr uint64_t STRIDE_B       = (uint64_t)(BN * 128) >> 4;
        constexpr uint64_t OFFSET_32      = 32 >> 4;

        for (int iter = 0; iter < NUM_ITERS; iter++) {
            const int stage = iter % STAGES;
            barrier::bar_wait(mbar_tma + stage * 8, (iter / STAGES) % 2);

            const uint64_t stage_inc = stage * STAGE_INC_DESC;
            
            const uint64_t d_sfa  = desc_sfa_base + stage_inc;
            const uint64_t d_sfb1 = desc_sfb1_base + stage_inc;
            const uint64_t d_sfb2 = desc_sfb2_base + stage_inc;

            #pragma unroll
            for (int k = 0; k < BK / Cfg::MMA_K; k++) {
                tmem::copy_scale(Cfg::TMEM_SFA  + k * 4, d_sfa  + k * 32);
                tmem::copy_scale(Cfg::TMEM_SFB1 + k * 4, d_sfb1 + k * 32);
                tmem::copy_scale(Cfg::TMEM_SFB2 + k * 4, d_sfb2 + k * 32);
            }

            uint64_t da  = desc_a_base  + stage_inc;
            uint64_t db1 = desc_b1_base + stage_inc;
            uint64_t db2 = desc_b2_base + stage_inc;

            for (int k1 = 0; k1 < BK / 256; k1++) {
                const int ksf = k1 * 4;
                const int tsfa  = tmem_sfa_base  + ksf * 4;
                const int tsfb1 = tmem_sfb1_base + ksf * 4;
                const int tsfb2 = tmem_sfb2_base + ksf * 4;

                {
                    const int acc = (k1 == 0 && iter == 0) ? 0 : 1;
                    mma::nvfp4_fill(da, db1, Cfg::MMA_IDESC, 0, tsfa, tsfb1, acc);
                    mma::nvfp4_lastuse(da, db2, Cfg::MMA_IDESC, BN, tsfa, tsfb2, acc);
                }
                
                #pragma unroll
                for (int k2 = 1; k2 < 4; k2++) {
                    da  += OFFSET_32;
                    db1 += OFFSET_32;
                    db2 += OFFSET_32;

                    const int ksf_inner = ksf + k2;
                    const int tsfa_i  = tmem_sfa_base  + ksf_inner * 4;
                    const int tsfb1_i = tmem_sfb1_base + ksf_inner * 4;
                    const int tsfb2_i = tmem_sfb2_base + ksf_inner * 4;

                    mma::nvfp4_fill(da, db1, Cfg::MMA_IDESC, 0, tsfa_i, tsfb1_i, 1);
                    mma::nvfp4_lastuse(da, db2, Cfg::MMA_IDESC, BN, tsfa_i, tsfb2_i, 1);
                }

                da  += (STRIDE_A - 3 * OFFSET_32);
                db1 += (STRIDE_B - 3 * OFFSET_32);
                db2 += (STRIDE_B - 3 * OFFSET_32);
            }

            tmem::commit(mbar_mma + stage * 8);
        }
        tmem::commit(mbar_done);
    }

    // Epilogue warps: TMEM -> global with silu fusion
    else if (tid < BM) {
        barrier::bar_wait(mbar_done, 0);
        tmem::fence_after_sync();

        float acc0_1[BN / 2], acc0_2[BN / 2];
        float acc1_1[BN / 2], acc1_2[BN / 2];
        
        tmem::load<BN>(acc0_1, warp * 32 + 0, 0);
        tmem::load<BN>(acc0_2, warp * 32 + 0, BN);
        tmem::load<BN>(acc1_1, warp * 32 + 16, 0);
        tmem::load<BN>(acc1_2, warp * 32 + 16, BN);
        
        tmem::wait_load();

        auto compute_store = [&](float* a1, float* a2, int m_offset) {
            #pragma unroll
            for (int i = 0; i < BN / 8; i++) {
                const int row = off_m + warp * 32 + m_offset + lane / 4;
                const int col = off_n + i * 8 + (lane % 4) * 2;

                const float r0 = math::silu(a1[i * 4 + 0]) * a2[i * 4 + 0];
                const float r1 = math::silu(a1[i * 4 + 1]) * a2[i * 4 + 1];
                store_cs(out + (row + 0) * N + col, __float22half2_rn({r0, r1}));

                const float r2 = math::silu(a1[i * 4 + 2]) * a2[i * 4 + 2];
                const float r3 = math::silu(a1[i * 4 + 3]) * a2[i * 4 + 3];
                store_cs(out + (row + 8) * N + col, __float22half2_rn({r2, r3}));
            }
        };

        compute_store(acc0_1, acc0_2, 0);
        compute_store(acc1_1, acc1_2, 16);

        asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");
        if (warp == 1)
            tmem::dealloc(0, Cfg::TMEM_COLS);
    }
}

template<int K, int BM, int BN, int BK, int Stages>
at::Tensor launch(
    const at::Tensor& A,  const at::Tensor& B1, const at::Tensor& B2,
    const at::Tensor& SFA, const at::Tensor& SFB1, const at::Tensor& SFB2,
    at::Tensor& C)
{
    using Cfg = KernelConfig<BM, BN, BK, Stages>;
    
    const int M = A.size(0);
    const int N = B1.size(0);

    CUtensorMap desc_a, desc_b1, desc_b2;
    tma::encode_desc(&desc_a,  reinterpret_cast<const char*>(A.data_ptr()),  M, K, BM, BK);
    tma::encode_desc(&desc_b1, reinterpret_cast<const char*>(B1.data_ptr()), N, K, BN, BK);
    tma::encode_desc(&desc_b2, reinterpret_cast<const char*>(B2.data_ptr()), N, K, BN, BK);

    const int grid = (M / BM) * (N / BN);
    const int smem = Cfg::STAGE_BYTES * Stages;

    auto kernel = dual_gemm_kernel<K, Cfg>;
    if (smem > 48000)
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    kernel<<<grid, Cfg::THREADS, smem>>>(
        desc_a, desc_b1, desc_b2,
        reinterpret_cast<const char*>(SFA.data_ptr()),
        reinterpret_cast<const char*>(SFB1.data_ptr()),
        reinterpret_cast<const char*>(SFB2.data_ptr()),
        reinterpret_cast<half*>(C.data_ptr()),
        M, N);

    return C;
}

at::Tensor dual_gemm(
    const at::Tensor& A,  const at::Tensor& B1, const at::Tensor& B2,
    const at::Tensor& SFA, const at::Tensor& SFB1, const at::Tensor& SFB2,
    at::Tensor& C)
{
    const int M = A.size(0);
    const int K = A.size(1) * 2;

    #define DISPATCH(Kval, BM, BN, BK, Stages) \
        return launch<Kval, BM, BN, BK, Stages>(A, B1, B2, SFA, SFB1, SFB2, C)

    switch (K) {
        case 7168: 
            if (M <= 256) { DISPATCH(7168, 128, 64, 256, 5); } 
            else          { DISPATCH(7168, 128, 128, 256, 4); }
        case 4096: DISPATCH(4096, 128, 64, 256, 5);
        case 2304: DISPATCH(2304, 128, 64, 256, 5);
        case 2048: DISPATCH(2048, 128, 64, 256, 5);
        case 1536: DISPATCH(1536, 128, 64, 256, 5);
        case 512:  DISPATCH(512,  128, 64, 256, 4);
        case 256:  DISPATCH(256,  128, 64, 256, 4);
        default:   TORCH_CHECK(false, "Unsupported K: ", K);
    }

    #undef DISPATCH
}

TORCH_LIBRARY(nvfp4_dual_gemm, m) {
    m.def("forward(Tensor A, Tensor B1, Tensor B2, Tensor SFA, Tensor SFB1, Tensor SFB2, Tensor(a!) C) -> Tensor");
    m.impl("forward", &dual_gemm);
}
'''

load_inline(
    "nvfp4_dual_gemm",
    cpp_sources="",
    cuda_sources=CUDA_SRC,
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        "-DNDEBUG",
        "-Xptxas=-v -O3",
    ],
    extra_ldflags=["-lcuda"],
)

forward = torch.ops.nvfp4_dual_gemm.forward
def custom_kernel(data: input_t) -> output_t:
    return forward(
        data[0], data[1], data[2],
        data[6], data[7], data[8],
        data[9]
    )
