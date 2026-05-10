// CRDT Merge — PTX-level comparison via CUDA device code
// Compares compiler-generated PTX vs handcoded-style approaches

#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define N_NODES 32
#define BLOOM_WORDS 94
#define N_RUNS 1000000

// ---- Device functions that compile to specific PTX patterns ----

// Standard max merge (compiler generates max.u64 PTX)
__device__ __forceinline__ void ptx_max_standard(volatile uint64_t *dst, const volatile uint64_t *src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (src[i] > dst[i]) ? src[i] : dst[i];
    }
}

// Bitwise OR merge (compiler generates or.b64 PTX)
__device__ __forceinline__ void ptx_or_standard(volatile uint64_t *dst, const volatile uint64_t *src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] |= src[i];
    }
}

// Using uint4 for 256-bit vectorized merge
__device__ __forceinline__ void ptx_or_vec4(volatile uint4 *dst, const volatile uint4 *src, int n_quads) {
    for (int i = 0; i < n_quads; i++) {
        uint4 d = dst[i];
        uint4 s = src[i];
        d.x |= s.x;
        d.y |= s.y;
        d.z |= s.z;
        d.w |= s.w;
        dst[i] = d;
    }
}

// Max merge using uint4
__device__ __forceinline__ void ptx_max_vec4(volatile uint4 *dst, const volatile uint4 *src, int n_quads) {
    for (int i = 0; i < n_quads; i++) {
        uint4 d = dst[i];
        uint4 s = src[i];
        d.x = (s.x > d.x) ? s.x : d.x;
        d.y = (s.y > d.y) ? s.y : d.y;
        d.z = (s.z > d.z) ? s.z : d.z;
        d.w = (s.w > d.w) ? s.w : d.w;
        dst[i] = d;
    }
}

// ---- Benchmark kernels ----

__global__ void bench_max_scalar(uint64_t *dst, const uint64_t *src, int n, int runs) {
    for (int r = 0; r < runs; r++) {
        ptx_max_standard(dst, src, n);
    }
}

__global__ void bench_max_vec4(uint4 *dst, const uint4 *src, int n, int runs) {
    for (int r = 0; r < runs; r++) {
        ptx_max_vec4(dst, src, n);
    }
}

__global__ void bench_or_scalar(uint64_t *dst, const uint64_t *src, int n, int runs) {
    for (int r = 0; r < runs; r++) {
        ptx_or_standard(dst, src, n);
    }
}

__global__ void bench_or_vec4(uint4 *dst, const uint4 *src, int n, int runs) {
    for (int r = 0; r < runs; r++) {
        ptx_or_vec4(dst, src, n);
    }
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err)); exit(1); } } while(0)

int main(void) {
    printf("=== CRDT Merge — PTX Instruction-Level Comparison ===\n");
    printf("GPU: ");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%s, SM %d.%d\n\n", prop.name, prop.major, prop.minor);

    // G-Counter: scalar vs vec4 max
    {
        int n = N_NODES;
        size_t bytes = n * sizeof(uint64_t);
        uint64_t *d_dst, *d_src;
        CUDA_CHECK(cudaMalloc(&d_dst, bytes));
        CUDA_CHECK(cudaMalloc(&d_src, bytes));
        
        uint64_t h[N_NODES];
        for (int i = 0; i < n; i++) h[i] = rand();
        CUDA_CHECK(cudaMemcpy(d_dst, h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_src, h, bytes, cudaMemcpyHostToDevice));

        // Warmup
        bench_max_scalar<<<1,1>>>(d_dst, d_src, n, 1000);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t start = now_ns();
        bench_max_scalar<<<1,1>>>(d_dst, d_src, n, N_RUNS);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t scalar_ns = now_ns() - start;

        // Vec4 version
        int n_quads = (n + 3) / 4;
        bench_max_vec4<<<1,1>>>((uint4*)d_dst, (const uint4*)d_src, n_quads, 1000);
        CUDA_CHECK(cudaDeviceSynchronize());

        start = now_ns();
        bench_max_vec4<<<1,1>>>((uint4*)d_dst, (const uint4*)d_src, n_quads, N_RUNS);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t vec4_ns = now_ns() - start;

        printf("  G-Counter max merge (%d elements):\n", n);
        printf("    Scalar (max.u64):    %8.1f ns/op\n", (double)scalar_ns / N_RUNS);
        printf("    Vectorized (uint4):  %8.1f ns/op\n", (double)vec4_ns / N_RUNS);
        printf("    Speedup: %.2fx\n\n", (double)scalar_ns / vec4_ns);

        CUDA_CHECK(cudaFree(d_dst));
        CUDA_CHECK(cudaFree(d_src));
    }

    // Bloom filter: scalar vs vec4 OR
    {
        int n = BLOOM_WORDS;
        size_t bytes = n * sizeof(uint64_t);
        uint64_t *d_dst, *d_src;
        CUDA_CHECK(cudaMalloc(&d_dst, bytes));
        CUDA_CHECK(cudaMalloc(&d_src, bytes));
        
        uint64_t *h = (uint64_t*)malloc(bytes);
        for (int i = 0; i < n; i++) h[i] = rand();
        CUDA_CHECK(cudaMemcpy(d_dst, h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_src, h, bytes, cudaMemcpyHostToDevice));
        free(h);

        bench_or_scalar<<<1,1>>>(d_dst, d_src, n, 1000);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t start = now_ns();
        bench_or_scalar<<<1,1>>>(d_dst, d_src, n, N_RUNS);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t scalar_ns = now_ns() - start;

        int n_quads = (n + 3) / 4;
        bench_or_vec4<<<1,1>>>((uint4*)d_dst, (const uint4*)d_src, n_quads, 1000);
        CUDA_CHECK(cudaDeviceSynchronize());

        start = now_ns();
        bench_or_vec4<<<1,1>>>((uint4*)d_dst, (const uint4*)d_src, n_quads, N_RUNS);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t vec4_ns = now_ns() - start;

        printf("  Bloom filter OR merge (%d words):\n", n);
        printf("    Scalar (or.b64):     %8.1f ns/op\n", (double)scalar_ns / N_RUNS);
        printf("    Vectorized (uint4):  %8.1f ns/op\n", (double)vec4_ns / N_RUNS);
        printf("    Speedup: %.2fx\n\n", (double)scalar_ns / vec4_ns);

        CUDA_CHECK(cudaFree(d_dst));
        CUDA_CHECK(cudaFree(d_src));
    }

    // Print PTX for inspection
    printf("--- PTX Instruction Analysis ---\n");
    printf("  max.u64: 1 register + 1 comparison + 1 store per element\n");
    printf("  or.b64:  1 register + 1 bitwise OR + 1 store per element\n");
    printf("  uint4:   4× bandwidth per instruction (256-bit)\n");
    printf("  Expected: or.b64 < max.u64 (OR is simpler than conditional max)\n");
    printf("  Expected: uint4 ~2-4x faster than scalar (memory bandwidth limited)\n");

    return 0;
}
