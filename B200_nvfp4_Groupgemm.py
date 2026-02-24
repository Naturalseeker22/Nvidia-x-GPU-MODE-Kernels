import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = r'''
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int MAX_GROUPS = 8;

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
    static constexpr int STAGE_BYTES = A_BYTES + B_BYTES + 2 * SF_BYTES;

    static constexpr int NUM_WARPS = BM / WARP_SIZE + 2;
    static constexpr int THREADS   = BM + 2 * WARP_SIZE;

    static constexpr int SF_COLS   = 4 * (BK / MMA_K);

    static constexpr int kNumEpilogueStages = (2 * BN + 2 * SF_COLS) > 512 ? 1 : 2;
    static constexpr int TMEM_ACCUM_COLS = kNumEpilogueStages * BN;
    static constexpr int TMEM_SFA  = TMEM_ACCUM_COLS;
    static constexpr int TMEM_SFB  = TMEM_SFA + SF_COLS;
    static constexpr int TMEM_COLS = 512;

    static constexpr uint32_t MMA_IDESC = (1U << 7) | (1U << 10) |
                                          ((uint32_t)BN >> 3 << 17) | (1U << 27);

    static constexpr int C_SWIZZLE_BYTES = 128;
    static constexpr int C_STORE_COLS = C_SWIZZLE_BYTES / 2;
    static constexpr int C_NUM_STORES = BN / C_STORE_COLS;
    static constexpr int C_STAGE_BYTES = BM * C_SWIZZLE_BYTES;
    static constexpr int C_NUM_STORE_STAGES = 2;

    static constexpr int INPUT_SMEM  = STAGE_BYTES * NUM_STAGES;
    static constexpr int OUTPUT_SMEM = C_STAGE_BYTES * C_NUM_STORE_STAGES;
    static constexpr int TOTAL_SMEM  = INPUT_SMEM + OUTPUT_SMEM;

    static constexpr int C_SMEM_OFFSET = INPUT_SMEM;
};


template<typename Cfg>
struct PipelineSmem {
    int base;

    struct Stage {
        int base;
        __device__ __forceinline__ int a()   const { return base; }
        __device__ __forceinline__ int b()   const { return base + Cfg::A_BYTES; }
        __device__ __forceinline__ int sfa() const { return base + Cfg::A_BYTES + Cfg::B_BYTES; }
        __device__ __forceinline__ int sfb() const { return base + Cfg::A_BYTES + Cfg::B_BYTES + Cfg::SF_BYTES; }
    };

    __device__ __forceinline__ Stage operator[](int i) const {
        return Stage{base + i * Cfg::STAGE_BYTES};
    }
};

namespace desc {

__device__ __forceinline__ uint64_t encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t matrix_header() {
    return (encode(1024) << 32) | (1ULL << 46) | (2ULL << 61);
}

__device__ __forceinline__ uint64_t scale_header() {
    return (encode(128) << 32) | (1ULL << 46);
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
        "@!p bra WAIT;\n"
        "}" :: "r"(addr), "r"(phase));
}

__device__ __forceinline__ void bar_arrive_tx(int addr, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void bar_arrive_remote(int local_addr, int remote_rank) {
    asm volatile(
        "{\n"
        ".reg .b32 remAddr;\n"
        "mapa.shared::cluster.u32 remAddr, %0, %1;\n"
        "mbarrier.arrive.shared::cluster.b64 _, [remAddr];\n"
        "}" :: "r"(local_addr), "r"(remote_rank) : "memory");
}

struct MBarrier {
    int addr;

    __device__ __forceinline__ static MBarrier at(int base, int idx) {
        return {base + idx * 8};
    }

    __device__ __forceinline__ void init(int count) const {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :: "r"(addr), "r"(count));
    }

    __device__ __forceinline__ void wait(int phase) const {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, 0x989680;\n"
            "@!p bra WAIT;\n"
            "}" :: "r"(addr), "r"(phase));
    }

    // Non-blocking try: arms hardware to monitor this barrier for faster subsequent waits
    __device__ __forceinline__ void try_wait(int phase) const {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, 0x989680;\n"
            "}" :: "r"(addr), "r"(phase));
    }



    __device__ __forceinline__ void arrive_tx(int bytes) const {
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(addr), "r"(bytes) : "memory");
    }

    __device__ __forceinline__ void arrive() const {
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                     :: "r"(addr) : "memory");
    }

    __device__ __forceinline__ void commit() const {
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                     :: "r"(addr) : "memory");
    }

};

} // namespace barrier

namespace cluster {

__device__ __forceinline__ int block_rank() {
    int rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

__device__ __forceinline__ void arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
}

__device__ __forceinline__ void wait_acquire() {
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

__device__ __forceinline__ void sync() {
    arrive_relaxed();
    wait_acquire();
}

} // namespace cluster

namespace tma {

constexpr uint64_t L2_NORMAL     = 0x1000000000000000ULL;
constexpr uint64_t L2_SdTREAMING  = 0x12F0000000000000ULL;
constexpr uint64_t L2_PERSISTENT = 0x14F0000000000000ULL;

__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void load_3d(int dst, const void* desc, int x, int y, int z,
                                        int mbar, uint64_t hint = L2_NORMAL) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
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

__device__ __forceinline__ void load_3d_mcast(int dst, const void* desc, int x, int y, int z,
                                              int mbar, int16_t cta_mask, uint64_t hint = L2_NORMAL) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
        :: "r"(dst), "l"(desc), "r"(x), "r"(y), "r"(z), "r"(mbar), "h"(cta_mask), "l"(hint) : "memory");
}

__device__ __forceinline__ void load_linear_mcast(int dst, const void* src, int bytes,
                                                   int mbar, int16_t cta_mask, uint64_t hint = L2_NORMAL) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4, %5;"
        :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar), "h"(cta_mask), "l"(hint));
}


void encode_desc(CUtensorMap* desc, const void* ptr,
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

__device__ __forceinline__ void store_fence() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void store_2d(const void* desc, int smem_addr,
                                          int x, int y, uint64_t hint) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%2, %3}], [%1], %4;"
        :: "l"(desc), "r"(smem_addr), "r"(x), "r"(y), "l"(hint) : "memory");
}

__device__ __forceinline__ void store_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template<int N>
__device__ __forceinline__ void store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ void prefetch_tensormap(const void* desc) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(desc) : "memory");
}

inline void create_c_tensormap(CUtensorMap* desc, void* ptr, int M, int N,
                                int box_m) {
    constexpr int STORE_COLS = 64;
    uint64_t dims[2]    = {(uint64_t)N, (uint64_t)M};
    uint64_t strides[1] = {(uint64_t)N * 2};
    uint32_t box[2]     = {STORE_COLS, (uint32_t)box_m};
    uint32_t elem[2]    = {1, 1};

    cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
                           ptr, dims, strides, box, elem,
                           CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}


__device__ __forceinline__ void replace_addr_in_smem(CUtensorMap* smem_desc,
                                                      void const* new_addr) {
    uint32_t s = smem_ptr_to_uint(smem_desc);
    uint64_t a = reinterpret_cast<uint64_t>(new_addr);
    asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
                 :: "r"(s), "l"(a) : "memory");
}

__device__ __forceinline__ void replace_dim_in_smem(CUtensorMap* smem_desc,
                                                     int dim_idx, uint32_t new_dim) {
    uint32_t s = smem_ptr_to_uint(smem_desc);
    switch (dim_idx) {
        case 0: asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" :: "r"(s), "r"(new_dim) : "memory"); break;
        case 1: asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;" :: "r"(s), "r"(new_dim) : "memory"); break;
        case 2: asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 2, %1;" :: "r"(s), "r"(new_dim) : "memory"); break;
    }
}

__device__ __forceinline__ void replace_stride_in_smem(CUtensorMap* smem_desc,
                                                        int stride_idx, uint64_t new_stride_bytes) {
    uint32_t s = smem_ptr_to_uint(smem_desc);
    uint64_t v = new_stride_bytes;
    switch (stride_idx) {
        case 0: asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;" :: "r"(s), "l"(v) : "memory"); break;
        case 1: asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 1, %1;" :: "r"(s), "l"(v) : "memory"); break;
    }
}

__device__ __forceinline__ void tensormap_release_fence() {
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}

__device__ __forceinline__ void fence_acquire(CUtensorMap const* gmem_desc) {
    uint64_t g = reinterpret_cast<uint64_t>(gmem_desc);
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                 :: "l"(g) : "memory");
}

__device__ __forceinline__ void update_a_desc(
    CUtensorMap* smem_desc, CUtensorMap* gmem_desc,
    const CUtensorMap* tmpl, const void* a_ptr,
    int M, int K, bool dynamic_k)
{
    uint4* dst = reinterpret_cast<uint4*>(smem_desc);
    uint4 const* src = reinterpret_cast<uint4 const*>(tmpl);
    #pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = src[i];

    replace_addr_in_smem(smem_desc, a_ptr);
    replace_dim_in_smem(smem_desc, 1, M);
    if (dynamic_k) {
        replace_dim_in_smem(smem_desc, 2, K / 256);
        replace_stride_in_smem(smem_desc, 0, (uint64_t)(K / 2));
    }

    uint4* gdst = reinterpret_cast<uint4*>(gmem_desc);
    uint4 const* ssrc = reinterpret_cast<uint4 const*>(smem_desc);
    #pragma unroll
    for (int i = 0; i < 8; i++) gdst[i] = ssrc[i];

    tensormap_release_fence();
    fence_acquire(gmem_desc);
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

__device__ __forceinline__ void fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void load_32x32b_x8(uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3,
                                                uint32_t& v4, uint32_t& v5, uint32_t& v6, uint32_t& v7, int row, int col) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3), "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
        : "r"((row << 16) | col));
}

__device__ __forceinline__ void load_32x32b_x32(uint32_t* v, int row, int col) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]),
          "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
          "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]),
          "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]),
          "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]),
          "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]),
          "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]),
          "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
        : "r"((row << 16) | col));
}

} // namespace tmem

__device__ __forceinline__ void st_shared_16b(uint8_t* ptr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))), "r"(v0), "r"(v1), "r"(v2), "r"(v3)
                 : "memory");
}

__device__ __forceinline__ uint32_t float2_to_half2_packed(float a, float b) {
    half2 h = __float22half2_rn(make_float2(a, b));
    return *reinterpret_cast<uint32_t*>(&h);
}

namespace mma {

__device__ __forceinline__ void nvfp4(uint64_t a, uint64_t b, uint32_t idesc,
                                      int d, int sfa, int sfb, int acc) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n"
        "}" :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc));
}


} // namespace mma

struct GroupPtrs {
    const char* A;
    const char* B;
    half* C;
    const char* SFA;
    const char* SFB;
};

struct alignas(128) GroupTensorMaps {
    CUtensorMap tma_a;
    CUtensorMap tma_b;
    CUtensorMap tma_c;
};

// All group A descriptors built at init; hot loop indexes by group_idx (no fences)
struct alignas(128) GmemTMapA {
    CUtensorMap tma_a[MAX_GROUPS];
};

GmemTMapA* get_gmem_tma_workspace(int num_ctas) {
    static GmemTMapA* s_ptr = nullptr;
    static int s_capacity = 0;
    if (num_ctas > s_capacity) {
        if (s_ptr) cudaFree(s_ptr);
        cudaMalloc(&s_ptr, num_ctas * sizeof(GmemTMapA));
        s_capacity = num_ctas;
    }
    return s_ptr;
}

struct KernelParams {
    GroupTensorMaps tensormaps[MAX_GROUPS];
    GroupPtrs   ptrs[MAX_GROUPS];
    int4        sizes[MAX_GROUPS];
    int         num_groups;
    int         total_ctas;
    CUtensorMap tma_a_template;
    GmemTMapA*  gmem_tma_a;
    bool        dynamic_k;
};

namespace tma {
__device__ __forceinline__ void init_all_a_descs(
    CUtensorMap* smem_desc, GmemTMapA* gmem_workspace,
    const CUtensorMap* tmpl, const KernelParams& params)
{
    for (int g = 0; g < params.num_groups; g++) {
        uint4* dst = reinterpret_cast<uint4*>(smem_desc);
        uint4 const* src = reinterpret_cast<uint4 const*>(tmpl);
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = src[i];

        replace_addr_in_smem(smem_desc, params.ptrs[g].A);
        replace_dim_in_smem(smem_desc, 1, params.sizes[g].x);  // M
        if (params.dynamic_k) {
            int K = params.sizes[g].z;
            replace_dim_in_smem(smem_desc, 2, K / 256);
            replace_stride_in_smem(smem_desc, 0, (uint64_t)(K / 2));
        }

        // SMEM → GMEM copy (no fence yet)
        uint4* gdst = reinterpret_cast<uint4*>(&gmem_workspace->tma_a[g]);
        uint4 const* ssrc = reinterpret_cast<uint4 const*>(smem_desc);
        #pragma unroll
        for (int i = 0; i < 8; i++) gdst[i] = ssrc[i];
    }

    // Single release fence for all groups
    tensormap_release_fence();

    // Acquire each group's descriptor
    for (int g = 0; g < params.num_groups; g++) {
        fence_acquire(&gmem_workspace->tma_a[g]);
    }
}

// Build A descriptors for groups [start_g, end_g) without fencing
__device__ __forceinline__ void init_a_descs_no_fence(
    CUtensorMap* smem_desc, GmemTMapA* gmem_workspace,
    const CUtensorMap* tmpl, const KernelParams& params,
    int start_g, int end_g)
{
    for (int g = start_g; g < end_g; g++) {
        uint4* dst = reinterpret_cast<uint4*>(smem_desc);
        uint4 const* src = reinterpret_cast<uint4 const*>(tmpl);
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = src[i];

        replace_addr_in_smem(smem_desc, params.ptrs[g].A);
        replace_dim_in_smem(smem_desc, 1, params.sizes[g].x);
        if (params.dynamic_k) {
            int K = params.sizes[g].z;
            replace_dim_in_smem(smem_desc, 2, K / 256);
            replace_stride_in_smem(smem_desc, 0, (uint64_t)(K / 2));
        }

        uint4* gdst = reinterpret_cast<uint4*>(&gmem_workspace->tma_a[g]);
        uint4 const* ssrc = reinterpret_cast<uint4 const*>(smem_desc);
        #pragma unroll
        for (int i = 0; i < 8; i++) gdst[i] = ssrc[i];
    }
}

// Release fence + acquire for groups [start_g, end_g)
__device__ __forceinline__ void fence_acquire_a_descs(
    GmemTMapA* gmem_workspace, int start_g, int end_g)
{
    tensormap_release_fence();
    for (int g = start_g; g < end_g; g++) {
        fence_acquire(&gmem_workspace->tma_a[g]);
    }
}

} // namespace tma

template<typename Cfg>
struct Scheduler {
    const KernelParams& params;
    int current_iter;

    int group_idx, tile_m, tile_n;
    int M, N, K;
    int off_m, off_n, num_k_iters;
    GroupPtrs ptrs;
    const void* tma_a;
    const void* tma_b;
    const void* tma_c;

    __device__ __forceinline__ Scheduler(const KernelParams& p)
        : params(p), current_iter(-1) {}

    __device__ __forceinline__ bool get_next_block() {
        const int bid = (++current_iter) * gridDim.x + blockIdx.x;
        if (bid >= params.total_ctas) return false;

        group_idx = 0;
        #pragma unroll
        for (int g = 1; g < MAX_GROUPS; g++) {
            if (g < params.num_groups && params.sizes[g].w <= bid) group_idx = g;
        }

        int local_bid = bid - params.sizes[group_idx].w;
        int4 ps = params.sizes[group_idx];
        M = ps.x; N = ps.y; K = ps.z;

        int tiles_m = (M + Cfg::BM - 1) / Cfg::BM;

        tile_m = local_bid % tiles_m;
        tile_n = local_bid / tiles_m;

        off_m = tile_m * Cfg::BM;
        off_n = tile_n * Cfg::BN;
        num_k_iters = K / Cfg::BK;

        ptrs = params.ptrs[group_idx];
        tma_b = &params.tensormaps[group_idx].tma_b;
        tma_c = &params.tensormaps[group_idx].tma_c;

        return true;
    }
};

// Grouped column-major: GROUP_M M-tiles per N-column for B L2 reuse
template<typename Cfg, int CLUSTER_N>
struct ClusterScheduler {
    const KernelParams& params;
    int current_iter;
    int rank;

    int group_idx, tile_m, tile_n;
    int M, N, K;
    int off_m, off_n, num_k_iters;
    GroupPtrs ptrs;
    const void* tma_a;
    const void* tma_b;
    const void* tma_c;

    __device__ __forceinline__ ClusterScheduler(const KernelParams& p, int rank_)
        : params(p), current_iter(-1), rank(rank_) {}

    __device__ __forceinline__ bool get_next_block() {
        const int cluster_bid = blockIdx.x / CLUSTER_N;
        const int num_clusters = gridDim.x / CLUSTER_N;
        const int cbid = (++current_iter) * num_clusters + cluster_bid;
        if (cbid >= params.total_ctas) return false;

        group_idx = 0;
        #pragma unroll
        for (int g = 1; g < MAX_GROUPS; g++) {
            if (g < params.num_groups && params.sizes[g].w <= cbid) group_idx = g;
        }

        int local_cbid = cbid - params.sizes[group_idx].w;
        int4 ps = params.sizes[group_idx];
        M = ps.x; N = ps.y; K = ps.z;

        int tiles_m = (M + Cfg::BM - 1) / Cfg::BM;
        int tiles_n = (N + Cfg::BN - 1) / Cfg::BN;
        int cluster_tiles_n = tiles_n / CLUSTER_N;

        constexpr int GROUP_M = 8;
        int group_m = min(tiles_m, GROUP_M);
        int tiles_per_super = group_m * cluster_tiles_n;

        int super_group = local_cbid / tiles_per_super;
        int within_super = local_cbid % tiles_per_super;

        int cluster_tile_n = within_super / group_m;
        int m_within = within_super % group_m;
        tile_m = super_group * group_m + m_within;
        tile_n = cluster_tile_n * CLUSTER_N + rank;

        off_m = tile_m * Cfg::BM;
        off_n = tile_n * Cfg::BN;
        num_k_iters = K / Cfg::BK;

        ptrs = params.ptrs[group_idx];
        tma_b = &params.tensormaps[group_idx].tma_b;
        tma_c = &params.tensormaps[group_idx].tma_c;

        return true;
    }
};


template<typename Cfg>
__global__ __launch_bounds__(Cfg::THREADS)
void group_gemm_kernel_v2(const __grid_constant__ KernelParams params)
{
    constexpr int BM = Cfg::BM;
    constexpr int BN = Cfg::BN;
    constexpr int BK = Cfg::BK;
    constexpr int STAGES = Cfg::NUM_STAGES;

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int warp = tid / Cfg::WARP_SIZE;

    int lo = 0, hi = params.num_groups;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (params.sizes[mid].w <= bid) lo = mid + 1; else hi = mid;
    }
    int group_idx = lo - 1;

    int local_bid = bid - params.sizes[group_idx].w;

    int4 ps = params.sizes[group_idx];
    const int M = ps.x;
    const int N = ps.y;
    const int K = ps.z;

    int tiles_m = (M + BM - 1) / BM;

    int tile_m = local_bid % tiles_m;
    int tile_n = local_bid / tiles_m;

    const int off_m = tile_m * BM;
    const int off_n = tile_n * BN;

    GroupPtrs ptrs = params.ptrs[group_idx];

    extern __shared__ __align__(1024) char smem[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem));

    CUtensorMap* smem_tma_a_desc = reinterpret_cast<CUtensorMap*>(smem + Cfg::TOTAL_SMEM);
    GmemTMapA* my_gmem_a = &params.gmem_tma_a[bid];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t barriers[STAGES * 2 + 1];
    const int mbar_tma  = static_cast<int>(__cvta_generic_to_shared(barriers));
    const int mbar_mma  = mbar_tma + STAGES * 8;
    const int mbar_done = mbar_mma + STAGES * 8;

    const int num_iters = K / BK;

    const void* tma_b = &params.tensormaps[group_idx].tma_b;
    const void* tma_c = &params.tensormaps[group_idx].tma_c;

    if (warp == 0) {
        if (barrier::elect_one()) {
            tma::update_a_desc(smem_tma_a_desc, &my_gmem_a->tma_a[0], &params.tma_a_template, ptrs.A, M, K, params.dynamic_k);
            tma::prefetch_tensormap(tma_b);
            tma::prefetch_tensormap(tma_c);
            #pragma unroll
            for (int i = 0; i < STAGES * 2 + 1; i++)
                barrier::bar_init(mbar_tma + i * 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }
    else if (warp == 1) {
        tmem::alloc(smem_base, Cfg::TMEM_COLS);
    }
    __syncthreads();

    const void* tma_a = &my_gmem_a->tma_a[0];

    // TMA warp
    if (warp == Cfg::NUM_WARPS - 2 && barrier::elect_one()) {
        const int sf_stride = K / 64;
        const int sf_m_off = (off_m / 128) * sf_stride;
        const int sf_n_off = (off_n / 128) * sf_stride;

        auto issue = [&](int iter, int stage) {
            const int mbar    = mbar_tma + stage * 8;
            const int s_a     = smem_base + stage * Cfg::STAGE_BYTES;
            const int s_b     = s_a + Cfg::A_BYTES;
            const int s_sfa   = s_b + Cfg::B_BYTES;
            const int s_sfb   = s_sfa + Cfg::SF_BYTES;

            const int z_coord = iter * BK / 256;

            tma::load_3d(s_a, tma_a, 0, off_m, z_coord, mbar, tma::L2_SdTREAMING);
            tma::load_3d(s_b, tma_b, 0, off_n, z_coord, mbar, tma::L2_SdTREAMING);

            tma::load_linear(s_sfa, ptrs.SFA + (sf_m_off + iter * BK / 64) * 512, Cfg::SF_BYTES, mbar, tma::L2_PERSISTENT);
            tma::load_linear(s_sfb, ptrs.SFB + (sf_n_off + iter * BK / 64) * 512, Cfg::SF_BYTES, mbar, tma::L2_PERSISTENT);

            barrier::bar_arrive_tx(mbar, Cfg::STAGE_BYTES);
        };

        #pragma unroll
        for (int i = 0; i < STAGES; i++)
            issue(i, i);

        for (int i = STAGES; i < num_iters; i++) {
            const int stage = i % STAGES;
            barrier::bar_wait(mbar_mma + stage * 8, (i / STAGES - 1) % 2);
            issue(i, stage);
        }
    }
    // MMA warp
    else if (warp == Cfg::NUM_WARPS - 1 && barrier::elect_one()) {
        const int tmem_sfa_base = Cfg::TMEM_SFA + (tile_m % (128 / BM)) * (BM / 32);
        const int tmem_sfb_base = Cfg::TMEM_SFB + (tile_n % (128 / BN)) * (BN / 32);

        const uint64_t m_desc_header = desc::matrix_header();
        const uint64_t s_desc_header = desc::scale_header();

        constexpr uint64_t STAGE_INC_DESC = Cfg::STAGE_BYTES >> 4;
        constexpr uint64_t STRIDE_A       = (uint64_t)(BM * 128) >> 4;
        constexpr uint64_t STRIDE_B       = (uint64_t)(BN * 128) >> 4;
        constexpr uint64_t OFFSET_32      = 32 >> 4;

        const uint64_t desc_a_base   = m_desc_header | ((uint64_t)smem_base >> 4);
        const uint64_t desc_b_base   = m_desc_header | ((uint64_t)(smem_base + Cfg::A_BYTES) >> 4);
        const uint64_t desc_sfa_base = s_desc_header | ((uint64_t)(smem_base + Cfg::A_BYTES + Cfg::B_BYTES) >> 4);
        const uint64_t desc_sfb_base = s_desc_header | ((uint64_t)(smem_base + Cfg::A_BYTES + Cfg::B_BYTES + Cfg::SF_BYTES) >> 4);

        int acc = 0;
        for (int iter = 0; iter < num_iters; iter++) {
            const int stage = iter % STAGES;
            barrier::bar_wait(mbar_tma + stage * 8, (iter / STAGES) % 2);

            const uint64_t stage_inc = stage * STAGE_INC_DESC;

            const uint64_t d_sfa = desc_sfa_base + stage_inc;
            const uint64_t d_sfb = desc_sfb_base + stage_inc;

            #pragma unroll
            for (int k = 0; k < BK / Cfg::MMA_K; k++) {
                tmem::copy_scale(Cfg::TMEM_SFA + k * 4, d_sfa + k * 32);
                tmem::copy_scale(Cfg::TMEM_SFB + k * 4, d_sfb + k * 32);
            }

            uint64_t da = desc_a_base + stage_inc;
            uint64_t db = desc_b_base + stage_inc;

            for (int k1 = 0; k1 < BK / 256; k1++) {
                const int ksf = k1 * 4;

                #pragma unroll
                for (int k2 = 0; k2 < 4; k2++) {
                    mma::nvfp4(da, db, Cfg::MMA_IDESC, 0,
                               tmem_sfa_base + (ksf + k2) * 4,
                               tmem_sfb_base + (ksf + k2) * 4, acc);
                    acc = 1;
                    da += OFFSET_32;
                    db += OFFSET_32;
                }
                da += STRIDE_A - 4 * OFFSET_32;
                db += STRIDE_B - 4 * OFFSET_32;
            }
            tmem::commit(mbar_mma + stage * 8);
        }
        tmem::commit(mbar_done);
    }
    // Epilogue warps
    else if (tid < BM) {
        const int lane = tid % Cfg::WARP_SIZE;
        const int valid_rows = min(M - off_m, (int)BM);
        const bool is_valid = (tid < valid_rows);

        constexpr int kSwizzleCDMode = 128;
        constexpr int kNumBankGroupBytes = 16;
        constexpr int STORE_BLOCK_N = 64;

        barrier::bar_wait(mbar_done, 0);
        tmem::fence_after_sync();

        int tma_store_stage = 0;

        #pragma unroll
        for (int store_idx = 0; store_idx < Cfg::C_NUM_STORES; store_idx++) {
            const int col_base = store_idx * STORE_BLOCK_N;
            const int tmem_row = warp * 32;

            uint8_t* c_smem = reinterpret_cast<uint8_t*>(smem + Cfg::C_SMEM_OFFSET + tma_store_stage * Cfg::C_STAGE_BYTES);
            const int c_smem_int = smem_base + Cfg::C_SMEM_OFFSET + tma_store_stage * Cfg::C_STAGE_BYTES;

            uint32_t v[32];
            tmem::load_32x32b_x32(v, tmem_row, col_base);

            if (tid == 0) tma::store_wait<Cfg::C_NUM_STORE_STAGES - 1>();
            asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");

            if (is_valid) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int swizzled_col = k ^ (lane % 8);
                    uint8_t* smem_ptr = c_smem + warp * 32 * kSwizzleCDMode +
                                        lane * kSwizzleCDMode + swizzled_col * kNumBankGroupBytes;

                    uint32_t h01 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 0]), *reinterpret_cast<float*>(&v[k * 8 + 1]));
                    uint32_t h23 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 2]), *reinterpret_cast<float*>(&v[k * 8 + 3]));
                    uint32_t h45 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 4]), *reinterpret_cast<float*>(&v[k * 8 + 5]));
                    uint32_t h67 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 6]), *reinterpret_cast<float*>(&v[k * 8 + 7]));

                    st_shared_16b(smem_ptr, h01, h23, h45, h67);
                }
            }

            tmem::load_32x32b_x32(v, tmem_row, col_base + 32);

            if (is_valid) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int swizzled_col = (4 + k) ^ (lane % 8);
                    uint8_t* smem_ptr = c_smem + warp * 32 * kSwizzleCDMode +
                                        lane * kSwizzleCDMode + swizzled_col * kNumBankGroupBytes;

                    uint32_t h01 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 0]), *reinterpret_cast<float*>(&v[k * 8 + 1]));
                    uint32_t h23 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 2]), *reinterpret_cast<float*>(&v[k * 8 + 3]));
                    uint32_t h45 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 4]), *reinterpret_cast<float*>(&v[k * 8 + 5]));
                    uint32_t h67 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 6]), *reinterpret_cast<float*>(&v[k * 8 + 7]));

                    st_shared_16b(smem_ptr, h01, h23, h45, h67);
                }
            }

            tma::store_fence();
            asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");

            if (tid == 0) {
                tma::store_2d(tma_c, c_smem_int, off_n + col_base, off_m, tma::L2_SdTREAMING);
                tma::store_commit();
            }

            tma_store_stage ^= 1;
        }

        if (tid == 0) tma::store_wait<0>();
        asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");
        if (warp == 1)
            tmem::dealloc(0, Cfg::TMEM_COLS);
    }
}

template<typename Cfg>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(Cfg::THREADS)
void group_gemm_kernel_v2_clustered(const __grid_constant__ KernelParams params)
{
    constexpr int BM = Cfg::BM;
    constexpr int BN = Cfg::BN;
    constexpr int BK = Cfg::BK;
    constexpr int STAGES = Cfg::NUM_STAGES;
    constexpr int CLUSTER_N = 2;

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int warp = tid / Cfg::WARP_SIZE;
    const int rank = cluster::block_rank();
    const int16_t self_mask  = (int16_t)(1 << rank);

    const int cluster_bid = bid / CLUSTER_N;

    int lo = 0, hi = params.num_groups;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (params.sizes[mid].w <= cluster_bid) lo = mid + 1; else hi = mid;
    }
    int group_idx = lo - 1;

    int local_cbid = cluster_bid - params.sizes[group_idx].w;

    int4 ps = params.sizes[group_idx];
    const int M = ps.x;
    const int N = ps.y;
    const int K = ps.z;

    int tiles_m = (M + BM - 1) / BM;
    int tiles_n = (N + BN - 1) / BN;

    int tile_m = local_cbid % tiles_m;
    int cluster_tile_n = local_cbid / tiles_m;

    int tile_n = cluster_tile_n * CLUSTER_N + rank;
    const int off_m = tile_m * BM;
    const int off_n = tile_n * BN;

    const int cluster_active = min(CLUSTER_N, tiles_n - cluster_tile_n * CLUSTER_N);
    const bool n_in_bounds = (rank < cluster_active);

    const int16_t a_mask = (rank == 0 && cluster_active > 1) ? (int16_t)((1 << cluster_active) - 1) : self_mask;

    GroupPtrs ptrs = params.ptrs[group_idx];
    const void* tma_b = &params.tensormaps[group_idx].tma_b;
    const void* tma_c = &params.tensormaps[group_idx].tma_c;

    extern __shared__ __align__(1024) char smem[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem));

    CUtensorMap* smem_tma_a_desc = reinterpret_cast<CUtensorMap*>(smem + Cfg::TOTAL_SMEM);
    GmemTMapA* my_gmem_a = &params.gmem_tma_a[bid];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t barriers[STAGES * 2 + 1];
    const int mbar_tma  = static_cast<int>(__cvta_generic_to_shared(barriers));
    const int mbar_mma  = mbar_tma + STAGES * 8;
    const int mbar_done = mbar_mma + STAGES * 8;

    const int num_iters = K / BK;

    if (warp == 0) {
        if (barrier::elect_one()) {
            tma::update_a_desc(smem_tma_a_desc, &my_gmem_a->tma_a[0], &params.tma_a_template, ptrs.A, M, K, params.dynamic_k);
            tma::prefetch_tensormap(tma_b);
            tma::prefetch_tensormap(tma_c);
            #pragma unroll
            for (int i = 0; i < STAGES; i++) {
                barrier::bar_init(mbar_tma + i * 8, 1);
                barrier::bar_init(mbar_mma + i * 8, (rank == 0) ? cluster_active : 1);
            }
            barrier::bar_init(mbar_done, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }
    else if (warp == 1) {
        if (n_in_bounds)
            tmem::alloc(smem_base, Cfg::TMEM_COLS);
    }

    cluster::sync();
    __syncthreads();

    if (!n_in_bounds) {
        return;
    }

    const void* tma_a = &my_gmem_a->tma_a[0];

    // TMA warp
    if (warp == Cfg::NUM_WARPS - 2 && barrier::elect_one()) {
        const int sf_stride = K / 64;
        const int sf_m_off = (off_m / 128) * sf_stride;
        const int sf_n_off = (off_n / 128) * sf_stride;

        auto issue = [&](int iter, int stage) {
            const int mbar = mbar_tma + stage * 8;
            const int s_a   = smem_base + stage * Cfg::STAGE_BYTES;
            const int s_b   = s_a + Cfg::A_BYTES;
            const int s_sfa = s_b + Cfg::B_BYTES;
            const int s_sfb = s_sfa + Cfg::SF_BYTES;

            const int z_coord = iter * BK / 256;

            // A+SFA: CTA0 multicasts; B+SFB: each CTA loads own
            if (rank == 0) {
                tma::load_3d_mcast(s_a, tma_a, 0, off_m, z_coord, mbar, a_mask, tma::L2_SdTREAMING);
                tma::load_linear_mcast(s_sfa, ptrs.SFA + (sf_m_off + iter * BK / 64) * 512,
                                       Cfg::SF_BYTES, mbar, a_mask, tma::L2_PERSISTENT);
            }

            tma::load_3d_mcast(s_b, tma_b, 0, off_n, z_coord, mbar, self_mask, tma::L2_SdTREAMING);
            tma::load_linear_mcast(s_sfb, ptrs.SFB + (sf_n_off + iter * BK / 64) * 512,
                                   Cfg::SF_BYTES, mbar, self_mask, tma::L2_PERSISTENT);

            barrier::bar_arrive_tx(mbar, Cfg::STAGE_BYTES);
        };

        #pragma unroll
        for (int i = 0; i < STAGES; i++)
            issue(i, i);

        for (int i = STAGES; i < num_iters; i++) {
            const int stage = i % STAGES;
            barrier::bar_wait(mbar_mma + stage * 8, (i / STAGES - 1) % 2);
            issue(i, stage);
        }
    }
    // MMA warp
    else if (warp == Cfg::NUM_WARPS - 1 && barrier::elect_one()) {
        const int tmem_sfa_base = Cfg::TMEM_SFA + (tile_m % (128 / BM)) * (BM / 32);
        const int tmem_sfb_base = Cfg::TMEM_SFB + (tile_n % (128 / BN)) * (BN / 32);

        const uint64_t m_desc_header = desc::matrix_header();
        const uint64_t s_desc_header = desc::scale_header();

        constexpr uint64_t STAGE_INC_DESC = Cfg::STAGE_BYTES >> 4;
        constexpr uint64_t STRIDE_A       = (uint64_t)(BM * 128) >> 4;
        constexpr uint64_t STRIDE_B       = (uint64_t)(BN * 128) >> 4;
        constexpr uint64_t OFFSET_32      = 32 >> 4;

        const uint64_t desc_a_base   = m_desc_header | ((uint64_t)smem_base >> 4);
        const uint64_t desc_b_base   = m_desc_header | ((uint64_t)(smem_base + Cfg::A_BYTES) >> 4);
        const uint64_t desc_sfa_base = s_desc_header | ((uint64_t)(smem_base + Cfg::A_BYTES + Cfg::B_BYTES) >> 4);
        const uint64_t desc_sfb_base = s_desc_header | ((uint64_t)(smem_base + Cfg::A_BYTES + Cfg::B_BYTES + Cfg::SF_BYTES) >> 4);

        int acc = 0;
        for (int iter = 0; iter < num_iters; iter++) {
            const int stage = iter % STAGES;
            barrier::bar_wait(mbar_tma + stage * 8, (iter / STAGES) % 2);

            const uint64_t stage_inc = stage * STAGE_INC_DESC;

            const uint64_t d_sfa = desc_sfa_base + stage_inc;
            const uint64_t d_sfb = desc_sfb_base + stage_inc;

            #pragma unroll
            for (int k = 0; k < BK / Cfg::MMA_K; k++) {
                tmem::copy_scale(Cfg::TMEM_SFA + k * 4, d_sfa + k * 32);
                tmem::copy_scale(Cfg::TMEM_SFB + k * 4, d_sfb + k * 32);
            }

            uint64_t da = desc_a_base + stage_inc;
            uint64_t db = desc_b_base + stage_inc;

            for (int k1 = 0; k1 < BK / 256; k1++) {
                const int ksf = k1 * 4;

                #pragma unroll
                for (int k2 = 0; k2 < 4; k2++) {
                    mma::nvfp4(da, db, Cfg::MMA_IDESC, 0,
                               tmem_sfa_base + (ksf + k2) * 4,
                               tmem_sfb_base + (ksf + k2) * 4, acc);
                    acc = 1;
                    da += OFFSET_32;
                    db += OFFSET_32;
                }
                da += STRIDE_A - 4 * OFFSET_32;
                db += STRIDE_B - 4 * OFFSET_32;
            }
            tmem::commit(mbar_mma + stage * 8);
            if (rank > 0 && cluster_active > 1) {
                barrier::bar_arrive_remote(mbar_mma + stage * 8, 0);
            }
        }
        tmem::commit(mbar_done);
    }
    // Epilogue warps
    else if (tid < BM) {
        const int lane = tid % Cfg::WARP_SIZE;
        const int valid_rows = min(M - off_m, (int)BM);
        const bool is_valid = (tid < valid_rows);

        constexpr int kSwizzleCDMode = 128;
        constexpr int kNumBankGroupBytes = 16;
        constexpr int STORE_BLOCK_N = 64;

        barrier::bar_wait(mbar_done, 0);
        tmem::fence_after_sync();

        int tma_store_stage = 0;

        #pragma unroll
        for (int store_idx = 0; store_idx < Cfg::C_NUM_STORES; store_idx++) {
            const int col_base = store_idx * STORE_BLOCK_N;
            const int tmem_row = warp * 32;

            uint8_t* c_smem = reinterpret_cast<uint8_t*>(smem + Cfg::C_SMEM_OFFSET + tma_store_stage * Cfg::C_STAGE_BYTES);
            const int c_smem_int = smem_base + Cfg::C_SMEM_OFFSET + tma_store_stage * Cfg::C_STAGE_BYTES;

            uint32_t v[32];
            tmem::load_32x32b_x32(v, tmem_row, col_base);

            if (tid == 0) tma::store_wait<Cfg::C_NUM_STORE_STAGES - 1>();
            asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");

            if (is_valid) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int swizzled_col = k ^ (lane % 8);
                    uint8_t* smem_ptr = c_smem + warp * 32 * kSwizzleCDMode +
                                        lane * kSwizzleCDMode + swizzled_col * kNumBankGroupBytes;

                    uint32_t h01 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 0]), *reinterpret_cast<float*>(&v[k * 8 + 1]));
                    uint32_t h23 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 2]), *reinterpret_cast<float*>(&v[k * 8 + 3]));
                    uint32_t h45 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 4]), *reinterpret_cast<float*>(&v[k * 8 + 5]));
                    uint32_t h67 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 6]), *reinterpret_cast<float*>(&v[k * 8 + 7]));

                    st_shared_16b(smem_ptr, h01, h23, h45, h67);
                }
            }

            tmem::load_32x32b_x32(v, tmem_row, col_base + 32);

            if (is_valid) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int swizzled_col = (4 + k) ^ (lane % 8);
                    uint8_t* smem_ptr = c_smem + warp * 32 * kSwizzleCDMode +
                                        lane * kSwizzleCDMode + swizzled_col * kNumBankGroupBytes;

                    uint32_t h01 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 0]), *reinterpret_cast<float*>(&v[k * 8 + 1]));
                    uint32_t h23 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 2]), *reinterpret_cast<float*>(&v[k * 8 + 3]));
                    uint32_t h45 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 4]), *reinterpret_cast<float*>(&v[k * 8 + 5]));
                    uint32_t h67 = float2_to_half2_packed(*reinterpret_cast<float*>(&v[k * 8 + 6]), *reinterpret_cast<float*>(&v[k * 8 + 7]));

                    st_shared_16b(smem_ptr, h01, h23, h45, h67);
                }
            }

            tma::store_fence();
            asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");

            if (tid == 0) {
                tma::store_2d(tma_c, c_smem_int, off_n + col_base, off_m, tma::L2_SdTREAMING);
                tma::store_commit();
            }

            tma_store_stage ^= 1;
        }

        if (tid == 0) tma::store_wait<0>();
        asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");
        if (warp == 1)
            tmem::dealloc(0, Cfg::TMEM_COLS);
    }
}

template<typename Cfg>
__global__ __launch_bounds__(Cfg::THREADS)
void group_gemm_kernel_v3_persistent(const __grid_constant__ KernelParams params)
{
    constexpr int BM = Cfg::BM;
    constexpr int BN = Cfg::BN;
    constexpr int BK = Cfg::BK;
    constexpr int STAGES = Cfg::NUM_STAGES;

    const int tid  = threadIdx.x;
    const int warp = tid / Cfg::WARP_SIZE;

    extern __shared__ __align__(1024) char smem[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem));

    PipelineSmem<Cfg> pipeline{smem_base};

    constexpr int kNumEpilogueStages = Cfg::kNumEpilogueStages;
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t barrier_storage[STAGES * 2 + kNumEpilogueStages * 2];
    __shared__ int descs_built_up_to;
    const int mbar_base = static_cast<int>(__cvta_generic_to_shared(barrier_storage));

    using MBarrier = barrier::MBarrier;
    MBarrier mbar_tma[STAGES], mbar_mma[STAGES];
    MBarrier mbar_tmem_full[kNumEpilogueStages];
    MBarrier mbar_tmem_empty[kNumEpilogueStages];

    #pragma unroll
    for (int i = 0; i < STAGES; i++) {
        mbar_tma[i] = MBarrier::at(mbar_base, i);
        mbar_mma[i] = MBarrier::at(mbar_base, STAGES + i);
    }
    #pragma unroll
    for (int i = 0; i < kNumEpilogueStages; i++) {
        mbar_tmem_full[i] = MBarrier::at(mbar_base, STAGES * 2 + i);
        mbar_tmem_empty[i] = MBarrier::at(mbar_base, STAGES * 2 + kNumEpilogueStages + i);
    }

    CUtensorMap* smem_tma_a_desc = reinterpret_cast<CUtensorMap*>(smem + Cfg::TOTAL_SMEM);
    GmemTMapA* my_gmem_a = &params.gmem_tma_a[blockIdx.x];

    if (warp == 0 && barrier::elect_one()) {
        // Async: build only group 0; MMA warp builds the rest during prologue
        tma::init_a_descs_no_fence(smem_tma_a_desc, my_gmem_a, &params.tma_a_template, params, 0, 1);
        tma::fence_acquire_a_descs(my_gmem_a, 0, 1);
        descs_built_up_to = 1;
        #pragma unroll
        for (int i = 0; i < STAGES; i++) {
            mbar_tma[i].init(1);
            mbar_mma[i].init(1);
        }
        #pragma unroll
        for (int i = 0; i < kNumEpilogueStages; i++) {
            mbar_tmem_full[i].init(1);
            mbar_tmem_empty[i].init(1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 1) {
        tmem::alloc(smem_base, Cfg::TMEM_COLS);
    }
    __syncthreads();

    int stage_idx = 0;
    int phase = 0;
    auto advance_stage = [&]() {
        stage_idx = (stage_idx + 1) % STAGES;
        phase ^= (stage_idx == 0);
    };

    Scheduler<Cfg> scheduler(params);

    // TMA warp
    if (warp == Cfg::NUM_WARPS - 2 && barrier::elect_one()) {
        const void* tma_a = nullptr;
        int sf_m_off, sf_n_off;
        int prev_group = -1;

        auto issue = [&](int k) {
            auto s = pipeline[stage_idx];
            const int z = k * BK / 256;
            tma::load_3d(s.a(), tma_a, 0, scheduler.off_m, z,
                         mbar_tma[stage_idx].addr, tma::L2_SdTREAMING);
            tma::load_3d(s.b(), scheduler.tma_b, 0, scheduler.off_n, z,
                         mbar_tma[stage_idx].addr, tma::L2_SdTREAMING);
            tma::load_linear(s.sfa(), scheduler.ptrs.SFA + (sf_m_off + k * BK / 64) * 512,
                             Cfg::SF_BYTES, mbar_tma[stage_idx].addr, tma::L2_PERSISTENT);
            tma::load_linear(s.sfb(), scheduler.ptrs.SFB + (sf_n_off + k * BK / 64) * 512,
                             Cfg::SF_BYTES, mbar_tma[stage_idx].addr, tma::L2_PERSISTENT);
            mbar_tma[stage_idx].arrive_tx(Cfg::STAGE_BYTES);
        };

        // First tile: prologue fills STAGES without waiting (mbar_mma satisfied at init)
        if (scheduler.get_next_block()) {
            if (scheduler.group_idx != prev_group) {
                while (*(volatile int*)&descs_built_up_to <= scheduler.group_idx) {}
                tma::prefetch_tensormap(scheduler.tma_b);
                tma::prefetch_tensormap(scheduler.tma_c);
                prev_group = scheduler.group_idx;
            }
            tma_a = &my_gmem_a->tma_a[scheduler.group_idx];
            const int sf_stride = scheduler.K / 64;
            sf_m_off = (scheduler.off_m / 128) * sf_stride;
            sf_n_off = (scheduler.off_n / 128) * sf_stride;

            #pragma unroll
            for (int i = 0; i < STAGES; i++, advance_stage())
                issue(i);

            for (int k = STAGES; k < scheduler.num_k_iters; k++, advance_stage()) {
                mbar_mma[stage_idx].wait(phase ^ 1);
                issue(k);
            }
        }

        // Subsequent tiles
        while (scheduler.get_next_block()) {
            if (scheduler.group_idx != prev_group) {
                while (*(volatile int*)&descs_built_up_to <= scheduler.group_idx) {}
                tma::prefetch_tensormap(scheduler.tma_b);
                tma::prefetch_tensormap(scheduler.tma_c);
                prev_group = scheduler.group_idx;
            }
            tma_a = &my_gmem_a->tma_a[scheduler.group_idx];
            const int sf_stride = scheduler.K / 64;
            sf_m_off = (scheduler.off_m / 128) * sf_stride;
            sf_n_off = (scheduler.off_n / 128) * sf_stride;

            for (int k = 0; k < scheduler.num_k_iters; k++, advance_stage()) {
                mbar_mma[stage_idx].wait(phase ^ 1);
                issue(k);
            }
        }
    }
    // MMA warp
    else if (warp == Cfg::NUM_WARPS - 1 && barrier::elect_one()) {
        // Async: build remaining group descriptors while TMA fills prologue
        if (params.num_groups > 1) {
            tma::init_a_descs_no_fence(smem_tma_a_desc, my_gmem_a, &params.tma_a_template, params, 1, params.num_groups);
            tma::fence_acquire_a_descs(my_gmem_a, 1, params.num_groups);
            __threadfence_block();
            descs_built_up_to = params.num_groups;
        }

        const uint64_t m_desc_header = desc::matrix_header();
        const uint64_t s_desc_header = desc::scale_header();

        constexpr uint64_t STAGE_INC = Cfg::STAGE_BYTES >> 4;
        constexpr uint64_t STRIDE_A = (uint64_t)(BM * 128) >> 4;
        constexpr uint64_t STRIDE_B = (uint64_t)(BN * 128) >> 4;
        constexpr uint64_t OFF_32 = 32 >> 4;

        auto s0 = pipeline[0];
        const uint64_t desc_a   = m_desc_header | ((uint64_t)s0.a() >> 4);
        const uint64_t desc_b   = m_desc_header | ((uint64_t)s0.b() >> 4);
        const uint64_t desc_sfa = s_desc_header | ((uint64_t)s0.sfa() >> 4);
        const uint64_t desc_sfb = s_desc_header | ((uint64_t)s0.sfb() >> 4);

        while (scheduler.get_next_block()) {
            const int accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const int accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            if (scheduler.current_iter >= kNumEpilogueStages) {
                mbar_tmem_empty[accum_stage_idx].wait(accum_phase_idx ^ 1);
            }

            const int tsfa = Cfg::TMEM_SFA + (scheduler.tile_m % (128 / BM)) * (BM / 32);
            const int tsfb = Cfg::TMEM_SFB + (scheduler.tile_n % (128 / BN)) * (BN / 32);
            const int accum_offset = accum_stage_idx * BN;

            int acc = 0;
            for (int k = 0; k < scheduler.num_k_iters; k++, advance_stage()) {
                mbar_tma[stage_idx].wait(phase);

                const uint64_t sinc = stage_idx * STAGE_INC;

                #pragma unroll
                for (int i = 0; i < BK / Cfg::MMA_K; i++) {
                    tmem::copy_scale(Cfg::TMEM_SFA + i * 4, desc_sfa + sinc + i * 32);
                    tmem::copy_scale(Cfg::TMEM_SFB + i * 4, desc_sfb + sinc + i * 32);
                }

                uint64_t da = desc_a + sinc;
                uint64_t db = desc_b + sinc;

                for (int k1 = 0; k1 < BK / 256; k1++) {
                    #pragma unroll
                    for (int k2 = 0; k2 < 4; k2++) {
                        mma::nvfp4(da, db, Cfg::MMA_IDESC, accum_offset,
                                   tsfa + (k1 * 4 + k2) * 4,
                                   tsfb + (k1 * 4 + k2) * 4, acc);
                        acc = 1;
                        da += OFF_32;
                        db += OFF_32;
                    }
                    da += STRIDE_A - 4 * OFF_32;
                    db += STRIDE_B - 4 * OFF_32;
                }

                mbar_mma[stage_idx].commit();
            }
            tmem::commit(mbar_tmem_full[accum_stage_idx].addr);
        }
    }
    // Epilogue warps
    else if (tid < BM) {
        const int lane = tid % Cfg::WARP_SIZE;

        constexpr int kSwizzleCDMode = 128;
        constexpr int kNumBankGroupBytes = 16;
        constexpr int STORE_BLOCK_N = 64;

        int tma_store_stage = 0;

        while (scheduler.get_next_block()) {
            const int accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const int accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            const int accum_offset = accum_stage_idx * BN;
            const int valid_rows = min(scheduler.M - scheduler.off_m, (int)BM);
            const bool is_valid = (tid < valid_rows);

            mbar_tmem_full[accum_stage_idx].wait(accum_phase_idx);
            tmem::fence_after_sync();

            #pragma unroll
            for (int store_idx = 0; store_idx < Cfg::C_NUM_STORES; store_idx++) {
                const int col_base = store_idx * STORE_BLOCK_N;

                const int tmem_row = warp * 32;

                uint8_t* c_smem = reinterpret_cast<uint8_t*>(smem + Cfg::C_SMEM_OFFSET + tma_store_stage * Cfg::C_STAGE_BYTES);
                const int c_smem_int = smem_base + Cfg::C_SMEM_OFFSET + tma_store_stage * Cfg::C_STAGE_BYTES;

                uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                tmem::load_32x32b_x8(v0, v1, v2, v3, v4, v5, v6, v7, tmem_row, accum_offset + col_base);

                if (tid == 0) tma::store_wait<Cfg::C_NUM_STORE_STAGES - 1>();
                asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");

                #pragma unroll
                for (int bg = 0; bg < 8; bg++) {
                    if (bg > 0) {
                        tmem::load_32x32b_x8(v0, v1, v2, v3, v4, v5, v6, v7, tmem_row, accum_offset + col_base + bg * 8);
                    }

                    if (is_valid) {
                        const int swizzled_col = bg ^ (lane % 8);
                        uint8_t* smem_ptr = c_smem + warp * 32 * kSwizzleCDMode +
                                            lane * kSwizzleCDMode + swizzled_col * kNumBankGroupBytes;

                        uint32_t h01 = float2_to_half2_packed(*reinterpret_cast<float*>(&v0), *reinterpret_cast<float*>(&v1));
                        uint32_t h23 = float2_to_half2_packed(*reinterpret_cast<float*>(&v2), *reinterpret_cast<float*>(&v3));
                        uint32_t h45 = float2_to_half2_packed(*reinterpret_cast<float*>(&v4), *reinterpret_cast<float*>(&v5));
                        uint32_t h67 = float2_to_half2_packed(*reinterpret_cast<float*>(&v6), *reinterpret_cast<float*>(&v7));

                        st_shared_16b(smem_ptr, h01, h23, h45, h67);
                    }
                }

                if (store_idx == Cfg::C_NUM_STORES - 1) {
                    tmem::fence_before_sync();
                    asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");
                    if (warp == 0 && barrier::elect_one()) {
                        mbar_tmem_empty[accum_stage_idx].arrive();
                    }
                }

                tma::store_fence();
                asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");

                if (tid == 0) {
                    tma::store_2d(scheduler.tma_c, c_smem_int,
                                  scheduler.off_n + col_base, scheduler.off_m, tma::L2_SdTREAMING);
                    tma::store_commit();
                }

                tma_store_stage ^= 1;
            }
        }

        if (tid == 0) tma::store_wait<0>();
    }

    __syncthreads();
    if (warp == 1) {
        tmem::dealloc(0, Cfg::TMEM_COLS);
    }
}

template<typename Cfg>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(Cfg::THREADS)
void group_gemm_kernel_v3_persistent_clustered(const __grid_constant__ KernelParams params)
{
    constexpr int BM = Cfg::BM;
    constexpr int BN = Cfg::BN;
    constexpr int BK = Cfg::BK;
    constexpr int STAGES = Cfg::NUM_STAGES;
    constexpr int CLUSTER_N = 2;

    const int tid  = threadIdx.x;
    const int warp = tid / Cfg::WARP_SIZE;

    const int rank = cluster::block_rank();
    const int16_t self_mask = (int16_t)(1 << rank);
    const int16_t a_mask = (int16_t)0x3;  // Both CTAs always active (tiles_n % 2 == 0)

    extern __shared__ __align__(1024) char smem[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem));

    PipelineSmem<Cfg> pipeline{smem_base};

    constexpr int kNumEpilogueStages = Cfg::kNumEpilogueStages;
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t barrier_storage[STAGES * 2 + kNumEpilogueStages * 2];
    __shared__ int descs_built_up_to;
    const int mbar_base = static_cast<int>(__cvta_generic_to_shared(barrier_storage));

    using MBarrier = barrier::MBarrier;
    MBarrier mbar_tma[STAGES], mbar_mma[STAGES];
    MBarrier mbar_tmem_full[kNumEpilogueStages];
    MBarrier mbar_tmem_empty[kNumEpilogueStages];

    #pragma unroll
    for (int i = 0; i < STAGES; i++) {
        mbar_tma[i] = MBarrier::at(mbar_base, i);
        mbar_mma[i] = MBarrier::at(mbar_base, STAGES + i);
    }
    #pragma unroll
    for (int i = 0; i < kNumEpilogueStages; i++) {
        mbar_tmem_full[i] = MBarrier::at(mbar_base, STAGES * 2 + i);
        mbar_tmem_empty[i] = MBarrier::at(mbar_base, STAGES * 2 + kNumEpilogueStages + i);
    }

    CUtensorMap* smem_tma_a_desc = reinterpret_cast<CUtensorMap*>(smem + Cfg::TOTAL_SMEM);
    GmemTMapA* my_gmem_a = &params.gmem_tma_a[blockIdx.x];

    if (warp == 0 && barrier::elect_one()) {
        // Async: build only group 0; MMA warp builds the rest during prologue
        tma::init_a_descs_no_fence(smem_tma_a_desc, my_gmem_a, &params.tma_a_template, params, 0, 1);
        tma::fence_acquire_a_descs(my_gmem_a, 0, 1);
        descs_built_up_to = 1;
        #pragma unroll
        for (int i = 0; i < STAGES; i++) {
            mbar_tma[i].init(1);
            mbar_mma[i].init(rank == 0 ? CLUSTER_N : 1);
        }
        #pragma unroll
        for (int i = 0; i < kNumEpilogueStages; i++) {
            mbar_tmem_full[i].init(1);
            mbar_tmem_empty[i].init(1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 1) {
        tmem::alloc(smem_base, Cfg::TMEM_COLS);
    }
    cluster::sync();
    __syncthreads();

    int stage_idx = 0;
    int phase = 0;
    auto advance_stage = [&]() {
        stage_idx = (stage_idx + 1) % STAGES;
        phase ^= (stage_idx == 0);
    };

    ClusterScheduler<Cfg, CLUSTER_N> scheduler(params, rank);

    // TMA warp
    if (warp == Cfg::NUM_WARPS - 2 && barrier::elect_one()) {
        const void* tma_a = nullptr;
        int sf_m_off, sf_n_off;
        int prev_group = -1;

        auto issue = [&](int k) {
            auto s = pipeline[stage_idx];
            const int z = k * BK / 256;
            if (rank == 0) {
                tma::load_3d_mcast(s.a(), tma_a, 0, scheduler.off_m, z,
                                   mbar_tma[stage_idx].addr, a_mask, tma::L2_SdTREAMING);
                tma::load_linear_mcast(s.sfa(), scheduler.ptrs.SFA + (sf_m_off + k * BK / 64) * 512,
                                       Cfg::SF_BYTES, mbar_tma[stage_idx].addr, a_mask, tma::L2_PERSISTENT);
            }
            tma::load_3d_mcast(s.b(), scheduler.tma_b, 0, scheduler.off_n, z,
                               mbar_tma[stage_idx].addr, self_mask, tma::L2_SdTREAMING);
            tma::load_linear_mcast(s.sfb(), scheduler.ptrs.SFB + (sf_n_off + k * BK / 64) * 512,
                                   Cfg::SF_BYTES, mbar_tma[stage_idx].addr, self_mask, tma::L2_PERSISTENT);
            mbar_tma[stage_idx].arrive_tx(Cfg::STAGE_BYTES);
        };

        // First tile: prologue fills STAGES without waiting (mbar_mma satisfied at init)
        if (scheduler.get_next_block()) {
            while (*(volatile int*)&descs_built_up_to <= scheduler.group_idx) {}
            tma::prefetch_tensormap(scheduler.tma_b);
            tma::prefetch_tensormap(scheduler.tma_c);
            prev_group = scheduler.group_idx;
            tma_a = &my_gmem_a->tma_a[scheduler.group_idx];
            const int sf_stride = scheduler.K / 64;
            sf_m_off = (scheduler.off_m / 128) * sf_stride;
            sf_n_off = (scheduler.off_n / 128) * sf_stride;

            #pragma unroll
            for (int i = 0; i < STAGES; i++, advance_stage())
                issue(i);

            for (int k = STAGES; k < scheduler.num_k_iters; k++, advance_stage()) {
                mbar_mma[stage_idx].wait(phase ^ 1);
                issue(k);
            }
        }

        // Subsequent tiles
        while (scheduler.get_next_block()) {
            if (scheduler.group_idx != prev_group) {
                while (*(volatile int*)&descs_built_up_to <= scheduler.group_idx) {}
                tma::prefetch_tensormap(scheduler.tma_b);
                tma::prefetch_tensormap(scheduler.tma_c);
                prev_group = scheduler.group_idx;
            }
            tma_a = &my_gmem_a->tma_a[scheduler.group_idx];
            const int sf_stride = scheduler.K / 64;
            sf_m_off = (scheduler.off_m / 128) * sf_stride;
            sf_n_off = (scheduler.off_n / 128) * sf_stride;

            for (int k = 0; k < scheduler.num_k_iters; k++, advance_stage()) {
                mbar_mma[stage_idx].wait(phase ^ 1);
                issue(k);
            }
        }
    }
    // MMA warp
    else if (warp == Cfg::NUM_WARPS - 1 && barrier::elect_one()) {
        // Async: build remaining group descriptors while TMA fills prologue
        if (params.num_groups > 1) {
            tma::init_a_descs_no_fence(smem_tma_a_desc, my_gmem_a, &params.tma_a_template, params, 1, params.num_groups);
            tma::fence_acquire_a_descs(my_gmem_a, 1, params.num_groups);
            __threadfence_block();
            descs_built_up_to = params.num_groups;
        }

        const uint64_t m_desc_header = desc::matrix_header();
        const uint64_t s_desc_header = desc::scale_header();

        constexpr uint64_t STAGE_INC = Cfg::STAGE_BYTES >> 4;
        constexpr uint64_t STRIDE_A = (uint64_t)(BM * 128) >> 4;
        constexpr uint64_t STRIDE_B = (uint64_t)(BN * 128) >> 4;
        constexpr uint64_t OFF_32 = 32 >> 4;

        auto s0 = pipeline[0];
        const uint64_t desc_a   = m_desc_header | ((uint64_t)s0.a() >> 4);
        const uint64_t desc_b   = m_desc_header | ((uint64_t)s0.b() >> 4);
        const uint64_t desc_sfa = s_desc_header | ((uint64_t)s0.sfa() >> 4);
        const uint64_t desc_sfb = s_desc_header | ((uint64_t)s0.sfb() >> 4);

        while (scheduler.get_next_block()) {
            const int accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const int accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            if (scheduler.current_iter >= kNumEpilogueStages) {
                mbar_tmem_empty[accum_stage_idx].wait(accum_phase_idx ^ 1);
            }

            const int tsfa = Cfg::TMEM_SFA + (scheduler.tile_m % (128 / BM)) * (BM / 32);
            const int tsfb = Cfg::TMEM_SFB + (scheduler.tile_n % (128 / BN)) * (BN / 32);
            const int accum_offset = accum_stage_idx * BN;

            int acc = 0;
            for (int k = 0; k < scheduler.num_k_iters; k++, advance_stage()) {
                mbar_tma[stage_idx].wait(phase);

                const uint64_t sinc = stage_idx * STAGE_INC;

                #pragma unroll
                for (int i = 0; i < BK / Cfg::MMA_K; i++) {
                    tmem::copy_scale(Cfg::TMEM_SFA + i * 4, desc_sfa + sinc + i * 32);
                    tmem::copy_scale(Cfg::TMEM_SFB + i * 4, desc_sfb + sinc + i * 32);
                }

                uint64_t da = desc_a + sinc;
                uint64_t db = desc_b + sinc;

                for (int k1 = 0; k1 < BK / 256; k1++) {
                    #pragma unroll
                    for (int k2 = 0; k2 < 4; k2++) {
                        mma::nvfp4(da, db, Cfg::MMA_IDESC, accum_offset,
                                   tsfa + (k1 * 4 + k2) * 4,
                                   tsfb + (k1 * 4 + k2) * 4, acc);
                        acc = 1;
                        da += OFF_32;
                        db += OFF_32;
                    }
                    da += STRIDE_A - 4 * OFF_32;
                    db += STRIDE_B - 4 * OFF_32;
                }

                mbar_mma[stage_idx].commit();
                if (rank > 0) {
                    barrier::bar_arrive_remote(mbar_mma[stage_idx].addr, 0);
                }
            }
            tmem::commit(mbar_tmem_full[accum_stage_idx].addr);
        }
    }
    // Epilogue warps
    else if (tid < BM) {
        const int lane = tid % Cfg::WARP_SIZE;

        constexpr int kSwizzleCDMode = 128;
        constexpr int kNumBankGroupBytes = 16;
        constexpr int STORE_BLOCK_N = 64;

        int tma_store_stage = 0;

        while (scheduler.get_next_block()) {
            const int accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const int accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            const int accum_offset = accum_stage_idx * BN;
            const int valid_rows = min(scheduler.M - scheduler.off_m, (int)BM);
            const bool is_valid = (tid < valid_rows);

            mbar_tmem_full[accum_stage_idx].wait(accum_phase_idx);
            tmem::fence_after_sync();

            #pragma unroll
            for (int store_idx = 0; store_idx < Cfg::C_NUM_STORES; store_idx++) {
                const int col_base = store_idx * STORE_BLOCK_N;

                const int tmem_row = warp * 32;

                uint8_t* c_smem = reinterpret_cast<uint8_t*>(smem + Cfg::C_SMEM_OFFSET + tma_store_stage * Cfg::C_STAGE_BYTES);
                const int c_smem_int = smem_base + Cfg::C_SMEM_OFFSET + tma_store_stage * Cfg::C_STAGE_BYTES;

                uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                tmem::load_32x32b_x8(v0, v1, v2, v3, v4, v5, v6, v7, tmem_row, accum_offset + col_base);

                if (tid == 0) tma::store_wait<Cfg::C_NUM_STORE_STAGES - 1>();
                asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");

                #pragma unroll
                for (int bg = 0; bg < 8; bg++) {
                    if (bg > 0) {
                        tmem::load_32x32b_x8(v0, v1, v2, v3, v4, v5, v6, v7, tmem_row, accum_offset + col_base + bg * 8);
                    }

                    if (is_valid) {
                        const int swizzled_col = bg ^ (lane % 8);
                        uint8_t* smem_ptr = c_smem + warp * 32 * kSwizzleCDMode +
                                            lane * kSwizzleCDMode + swizzled_col * kNumBankGroupBytes;

                        uint32_t h01 = float2_to_half2_packed(*reinterpret_cast<float*>(&v0), *reinterpret_cast<float*>(&v1));
                        uint32_t h23 = float2_to_half2_packed(*reinterpret_cast<float*>(&v2), *reinterpret_cast<float*>(&v3));
                        uint32_t h45 = float2_to_half2_packed(*reinterpret_cast<float*>(&v4), *reinterpret_cast<float*>(&v5));
                        uint32_t h67 = float2_to_half2_packed(*reinterpret_cast<float*>(&v6), *reinterpret_cast<float*>(&v7));

                        st_shared_16b(smem_ptr, h01, h23, h45, h67);
                    }
                }

                if (store_idx == Cfg::C_NUM_STORES - 1) {
                    tmem::fence_before_sync();
                    asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");
                    if (warp == 0 && barrier::elect_one()) {
                        mbar_tmem_empty[accum_stage_idx].arrive();
                    }
                }

                tma::store_fence();
                asm volatile("bar.sync 1, %0;" :: "r"(BM) : "memory");

                if (tid == 0) {
                    tma::store_2d(scheduler.tma_c, c_smem_int,
                                  scheduler.off_n + col_base, scheduler.off_m, tma::L2_SdTREAMING);
                    tma::store_commit();
                }

                tma_store_stage ^= 1;
            }
        }

        if (tid == 0) tma::store_wait<0>();
    }

    __syncthreads();
    if (warp == 1) {
        tmem::dealloc(0, Cfg::TMEM_COLS);
    }
}


void group_gemm_v2(
    const at::Tensor& a_ptrs,
    const at::Tensor& b_ptrs,
    const at::Tensor& c_ptrs,
    const at::Tensor& sfa_ptrs,
    const at::Tensor& sfb_ptrs,
    const at::Tensor& problem_m,
    const at::Tensor& problem_n,
    const at::Tensor& problem_k,
    int64_t num_groups)
{
    constexpr int BM = 128, BN = 128, BK = 256, Stages = 5;
    using Cfg = KernelConfig<BM, BN, BK, Stages>;

    KernelParams params;
    params.num_groups = num_groups;
    params.total_ctas = 0;

    int cta_offset = 0;
    for (int g = 0; g < num_groups; g++) {
        int M = problem_m.data_ptr<int>()[g];
        int N = problem_n.data_ptr<int>()[g];
        int K = problem_k.data_ptr<int>()[g];

        const char* a_ptr = reinterpret_cast<const char*>(a_ptrs.data_ptr<int64_t>()[g]);
        const char* b_ptr = reinterpret_cast<const char*>(b_ptrs.data_ptr<int64_t>()[g]);
        half* c_ptr = reinterpret_cast<half*>(c_ptrs.data_ptr<int64_t>()[g]);
        const char* sfa_ptr = reinterpret_cast<const char*>(sfa_ptrs.data_ptr<int64_t>()[g]);
        const char* sfb_ptr = reinterpret_cast<const char*>(sfb_ptrs.data_ptr<int64_t>()[g]);

        tma::encode_desc(&params.tensormaps[g].tma_b, b_ptr, N, K, BN, BK);
        tma::create_c_tensormap(&params.tensormaps[g].tma_c, c_ptr, M, N, BM);

        params.ptrs[g] = {a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr};
        params.sizes[g] = make_int4(M, N, K, cta_offset);

        int tiles_m = (M + BM - 1) / BM;
        int tiles_n = (N + BN - 1) / BN;
        cta_offset += tiles_m * tiles_n;
    }
    params.total_ctas = cta_offset;

    int K0 = problem_k.data_ptr<int>()[0];
    const char* a0 = reinterpret_cast<const char*>(a_ptrs.data_ptr<int64_t>()[0]);
    tma::encode_desc(&params.tma_a_template, a0, 128, K0, BM, BK);
    params.gmem_tma_a = get_gmem_tma_workspace(params.total_ctas);
    params.dynamic_k = false;
    for (int g = 1; g < num_groups; g++) {
        if (problem_k.data_ptr<int>()[g] != K0) { params.dynamic_k = true; break; }
    }

    const int smem = Cfg::TOTAL_SMEM + 128;
    auto kernel = group_gemm_kernel_v2<Cfg>;
    if (smem > 48000)
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    kernel<<<params.total_ctas, Cfg::THREADS, smem>>>(params);
}

void group_gemm_clustered(
    const at::Tensor& a_ptrs,
    const at::Tensor& b_ptrs,
    const at::Tensor& c_ptrs,
    const at::Tensor& sfa_ptrs,
    const at::Tensor& sfb_ptrs,
    const at::Tensor& problem_m,
    const at::Tensor& problem_n,
    const at::Tensor& problem_k,
    int64_t num_groups)
{
    constexpr int BM = 128, BN = 128, BK = 256, Stages = 5;
    constexpr int CLUSTER_N = 2;
    using Cfg = KernelConfig<BM, BN, BK, Stages>;

    KernelParams params;
    params.num_groups = num_groups;
    params.total_ctas = 0;

    int cluster_tile_offset = 0;
    for (int g = 0; g < num_groups; g++) {
        int M = problem_m.data_ptr<int>()[g];
        int N = problem_n.data_ptr<int>()[g];
        int K = problem_k.data_ptr<int>()[g];

        const char* a_ptr = reinterpret_cast<const char*>(a_ptrs.data_ptr<int64_t>()[g]);
        const char* b_ptr = reinterpret_cast<const char*>(b_ptrs.data_ptr<int64_t>()[g]);
        half* c_ptr = reinterpret_cast<half*>(c_ptrs.data_ptr<int64_t>()[g]);
        const char* sfa_ptr = reinterpret_cast<const char*>(sfa_ptrs.data_ptr<int64_t>()[g]);
        const char* sfb_ptr = reinterpret_cast<const char*>(sfb_ptrs.data_ptr<int64_t>()[g]);

        tma::encode_desc(&params.tensormaps[g].tma_b, b_ptr, N, K, BN, BK);
        tma::create_c_tensormap(&params.tensormaps[g].tma_c, c_ptr, M, N, BM);

        params.ptrs[g] = {a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr};
        params.sizes[g] = make_int4(M, N, K, cluster_tile_offset);

        int tiles_m = (M + BM - 1) / BM;
        int tiles_n = (N + BN - 1) / BN;
        int cluster_tiles_n = (tiles_n + CLUSTER_N - 1) / CLUSTER_N;
        cluster_tile_offset += tiles_m * cluster_tiles_n;
    }
    int total_cluster_tiles = cluster_tile_offset;
    int total_ctas = total_cluster_tiles * CLUSTER_N;
    params.total_ctas = total_cluster_tiles;

    int K0 = problem_k.data_ptr<int>()[0];
    const char* a0 = reinterpret_cast<const char*>(a_ptrs.data_ptr<int64_t>()[0]);
    tma::encode_desc(&params.tma_a_template, a0, 128, K0, BM, BK);
    params.gmem_tma_a = get_gmem_tma_workspace(total_ctas);
    params.dynamic_k = false;
    for (int g = 1; g < num_groups; g++) {
        if (problem_k.data_ptr<int>()[g] != K0) { params.dynamic_k = true; break; }
    }

    const int smem = Cfg::TOTAL_SMEM + 128;
    auto kernel = group_gemm_kernel_v2_clustered<Cfg>;
    if (smem > 48000)
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    cudaLaunchConfig_t launch_config = {};
    launch_config.blockDim = {(unsigned int)Cfg::THREADS, 1, 1};
    launch_config.gridDim = {(unsigned int)total_ctas, 1, 1};
    launch_config.dynamicSmemBytes = smem;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {CLUSTER_N, 1, 1};
    launch_config.attrs = attrs;
    launch_config.numAttrs = 1;

    auto status = cudaLaunchKernelEx(&launch_config, kernel, params);
    if (status != cudaSuccess) {
        TORCH_CHECK(false, "cudaLaunchKernelEx failed: ", cudaGetErrorString(status));
    }
}

void group_gemm_v3_persistent(
    const at::Tensor& a_ptrs,
    const at::Tensor& b_ptrs,
    const at::Tensor& c_ptrs,
    const at::Tensor& sfa_ptrs,
    const at::Tensor& sfb_ptrs,
    const at::Tensor& problem_m,
    const at::Tensor& problem_n,
    const at::Tensor& problem_k,
    int64_t num_groups)
{
    constexpr int BM = 128, BN = 128, BK = 256, Stages = 5;
    using Cfg = KernelConfig<BM, BN, BK, Stages>;

    KernelParams params;
    params.num_groups = num_groups;
    params.total_ctas = 0;

    int cta_offset = 0;
    for (int g = 0; g < num_groups; g++) {
        int M = problem_m.data_ptr<int>()[g];
        int N = problem_n.data_ptr<int>()[g];
        int K = problem_k.data_ptr<int>()[g];

        const char* a_ptr = reinterpret_cast<const char*>(a_ptrs.data_ptr<int64_t>()[g]);
        const char* b_ptr = reinterpret_cast<const char*>(b_ptrs.data_ptr<int64_t>()[g]);
        half* c_ptr = reinterpret_cast<half*>(c_ptrs.data_ptr<int64_t>()[g]);
        const char* sfa_ptr = reinterpret_cast<const char*>(sfa_ptrs.data_ptr<int64_t>()[g]);
        const char* sfb_ptr = reinterpret_cast<const char*>(sfb_ptrs.data_ptr<int64_t>()[g]);

        tma::encode_desc(&params.tensormaps[g].tma_b, b_ptr, N, K, BN, BK);
        tma::create_c_tensormap(&params.tensormaps[g].tma_c, c_ptr, M, N, BM);

        params.ptrs[g] = {a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr};
        params.sizes[g] = make_int4(M, N, K, cta_offset);

        int tiles_m = (M + BM - 1) / BM;
        int tiles_n = (N + BN - 1) / BN;
        cta_offset += tiles_m * tiles_n;
    }
    params.total_ctas = cta_offset;

    constexpr int NUM_SMS = 148;
    int num_ctas = min(params.total_ctas, NUM_SMS);

    int K0 = problem_k.data_ptr<int>()[0];
    const char* a0 = reinterpret_cast<const char*>(a_ptrs.data_ptr<int64_t>()[0]);
    tma::encode_desc(&params.tma_a_template, a0, 128, K0, BM, BK);
    params.gmem_tma_a = get_gmem_tma_workspace(num_ctas);
    params.dynamic_k = false;
    for (int g = 1; g < num_groups; g++) {
        if (problem_k.data_ptr<int>()[g] != K0) { params.dynamic_k = true; break; }
    }

    const int smem = Cfg::TOTAL_SMEM + 128;
    auto kernel = group_gemm_kernel_v3_persistent<Cfg>;
    if (smem > 48000)
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    kernel<<<num_ctas, Cfg::THREADS, smem>>>(params);
}

void group_gemm_v3_persistent_clustered(
    const at::Tensor& a_ptrs,
    const at::Tensor& b_ptrs,
    const at::Tensor& c_ptrs,
    const at::Tensor& sfa_ptrs,
    const at::Tensor& sfb_ptrs,
    const at::Tensor& problem_m,
    const at::Tensor& problem_n,
    const at::Tensor& problem_k,
    int64_t num_groups)
{
    constexpr int BM = 128, BN = 128, BK = 256, Stages = 5;
    constexpr int CLUSTER_N = 2;
    using Cfg = KernelConfig<BM, BN, BK, Stages>;

    KernelParams params;
    params.num_groups = num_groups;

    int cluster_tile_offset = 0;
    for (int g = 0; g < num_groups; g++) {
        int M = problem_m.data_ptr<int>()[g];
        int N = problem_n.data_ptr<int>()[g];
        int K = problem_k.data_ptr<int>()[g];

        const char* a_ptr = reinterpret_cast<const char*>(a_ptrs.data_ptr<int64_t>()[g]);
        const char* b_ptr = reinterpret_cast<const char*>(b_ptrs.data_ptr<int64_t>()[g]);
        half* c_ptr = reinterpret_cast<half*>(c_ptrs.data_ptr<int64_t>()[g]);
        const char* sfa_ptr = reinterpret_cast<const char*>(sfa_ptrs.data_ptr<int64_t>()[g]);
        const char* sfb_ptr = reinterpret_cast<const char*>(sfb_ptrs.data_ptr<int64_t>()[g]);

        tma::encode_desc(&params.tensormaps[g].tma_b, b_ptr, N, K, BN, BK);
        tma::create_c_tensormap(&params.tensormaps[g].tma_c, c_ptr, M, N, BM);

        params.ptrs[g] = {a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr};
        params.sizes[g] = make_int4(M, N, K, cluster_tile_offset);

        int tiles_m = (M + BM - 1) / BM;
        int tiles_n = (N + BN - 1) / BN;
        cluster_tile_offset += tiles_m * (tiles_n / CLUSTER_N);
    }
    int total_cluster_tiles = cluster_tile_offset;
    params.total_ctas = total_cluster_tiles;

    constexpr int NUM_SMS = 148;
    int num_clusters = min(total_cluster_tiles, NUM_SMS / CLUSTER_N);
    int num_ctas = num_clusters * CLUSTER_N;

    int K0 = problem_k.data_ptr<int>()[0];
    const char* a0 = reinterpret_cast<const char*>(a_ptrs.data_ptr<int64_t>()[0]);
    tma::encode_desc(&params.tma_a_template, a0, 128, K0, BM, BK);
    params.gmem_tma_a = get_gmem_tma_workspace(num_ctas);
    params.dynamic_k = false;
    for (int g = 1; g < num_groups; g++) {
        if (problem_k.data_ptr<int>()[g] != K0) { params.dynamic_k = true; break; }
    }

    const int smem = Cfg::TOTAL_SMEM + 128;
    auto kernel = group_gemm_kernel_v3_persistent_clustered<Cfg>;
    if (smem > 48000)
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    cudaLaunchConfig_t launch_config = {};
    launch_config.blockDim = {(unsigned int)Cfg::THREADS, 1, 1};
    launch_config.gridDim = {(unsigned int)num_ctas, 1, 1};
    launch_config.dynamicSmemBytes = smem;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {CLUSTER_N, 1, 1};
    launch_config.attrs = attrs;
    launch_config.numAttrs = 1;

    auto status = cudaLaunchKernelEx(&launch_config, kernel, params);
    if (status != cudaSuccess) {
        TORCH_CHECK(false, "cudaLaunchKernelEx failed: ", cudaGetErrorString(status));
    }
}


TORCH_LIBRARY(group_gemm, m) {
    m.def("forward_cutlass(Tensor a_ptrs, Tensor b_ptrs, Tensor c_ptrs, Tensor sfa_ptrs, Tensor sfb_ptrs, "
          "Tensor problem_m, Tensor problem_n, Tensor problem_k, int num_groups) -> ()");
    m.impl("forward_cutlass", &group_gemm_v2);
    m.def("forward_clustered(Tensor a_ptrs, Tensor b_ptrs, Tensor c_ptrs, Tensor sfa_ptrs, Tensor sfb_ptrs, "
          "Tensor problem_m, Tensor problem_n, Tensor problem_k, int num_groups) -> ()");
    m.impl("forward_clustered", &group_gemm_clustered);
    m.def("forward_v3_persistent(Tensor a_ptrs, Tensor b_ptrs, Tensor c_ptrs, Tensor sfa_ptrs, Tensor sfb_ptrs, "
          "Tensor problem_m, Tensor problem_n, Tensor problem_k, int num_groups) -> ()");
    m.impl("forward_v3_persistent", &group_gemm_v3_persistent);
    m.def("forward_v3_persistent_clustered(Tensor a_ptrs, Tensor b_ptrs, Tensor c_ptrs, Tensor sfa_ptrs, Tensor sfb_ptrs, "
          "Tensor problem_m, Tensor problem_n, Tensor problem_k, int num_groups) -> ()");
    m.impl("forward_v3_persistent_clustered", &group_gemm_v3_persistent_clustered);
}
'''

load_inline(
    "group_gemm",
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

forward_cutlass = torch.ops.group_gemm.forward_cutlass
forward_clustered = torch.ops.group_gemm.forward_clustered
forward_v3_persistent = torch.ops.group_gemm.forward_v3_persistent
forward_v3_persistent_clustered = torch.ops.group_gemm.forward_v3_persistent_clustered


def custom_kernel(data: input_t) -> output_t:
    """Main entry point - called by eval.py."""
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    num_groups = len(problem_sizes)

    a_ptrs, b_ptrs, c_ptrs, sfa_ptrs, sfb_ptrs = [], [], [], [], []
    m_list, n_list, k_list = [], [], []

    for g, ((a, b, c), (sfa, sfb), (m, n, k, l)) in enumerate(
        zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes)
    ):
        a_ptrs.append(a.data_ptr())
        b_ptrs.append(b.data_ptr())
        c_ptrs.append(c.data_ptr())
        sfa_ptrs.append(sfa.data_ptr())
        sfb_ptrs.append(sfb.data_ptr())
        m_list.append(m)
        n_list.append(n)
        k_list.append(k)

    a_ptrs_t = torch.tensor(a_ptrs, dtype=torch.int64, device="cpu")
    b_ptrs_t = torch.tensor(b_ptrs, dtype=torch.int64, device="cpu")
    c_ptrs_t = torch.tensor(c_ptrs, dtype=torch.int64, device="cpu")
    sfa_ptrs_t = torch.tensor(sfa_ptrs, dtype=torch.int64, device="cpu")
    sfb_ptrs_t = torch.tensor(sfb_ptrs, dtype=torch.int64, device="cpu")
    problem_m_t = torch.tensor(m_list, dtype=torch.int32, device="cpu")
    problem_n_t = torch.tensor(n_list, dtype=torch.int32, device="cpu")
    problem_k_t = torch.tensor(k_list, dtype=torch.int32, device="cpu")

    BN = 128
    CLUSTER_N = 2
    can_cluster = all((n + BN - 1) // BN % CLUSTER_N == 0 for _, n, _, _ in problem_sizes)
    if num_groups > 4 and can_cluster and max(k_list) > 2048:
        forward_v3_persistent_clustered(
            a_ptrs_t, b_ptrs_t, c_ptrs_t, sfa_ptrs_t, sfb_ptrs_t,
            problem_m_t, problem_n_t, problem_k_t, num_groups
        )
    elif num_groups > 4:
        forward_v3_persistent(
            a_ptrs_t, b_ptrs_t, c_ptrs_t, sfa_ptrs_t, sfb_ptrs_t,
            problem_m_t, problem_n_t, problem_k_t, num_groups
        )
    elif max(k_list) > 2048:
        forward_clustered(
            a_ptrs_t, b_ptrs_t, c_ptrs_t, sfa_ptrs_t, sfb_ptrs_t,
            problem_m_t, problem_n_t, problem_k_t, num_groups
        )
    else:
        forward_cutlass(
            a_ptrs_t, b_ptrs_t, c_ptrs_t, sfa_ptrs_t, sfb_ptrs_t,
            problem_m_t, problem_n_t, problem_k_t, num_groups
        )

    return [abc_tensors[i][2] for i in range(num_groups)]
