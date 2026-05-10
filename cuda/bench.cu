// CRDT Merge Micro-Benchmark — CUDA
// Parallel merge of CRDT states across thousands of fleet nodes
// Key insight: CRDT merge is element-wise independent → perfect GPU parallelism

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define N_NODES 32
#define N_CONSTRAINTS 256
#define BLOOM_WORDS 94
#define SKETCH_D 7
#define SKETCH_W 1000

// ---- CUDA Kernels ----

__global__ void gcounter_merge_kernel(
    uint64_t * __restrict__ dst_counts, uint64_t * __restrict__ dst_violations,
    const uint64_t * __restrict__ src_counts, const uint64_t * __restrict__ src_violations,
    int n_nodes, int n_merges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_nodes * n_merges) {
        int node = idx % n_nodes;
        int merge = idx / n_nodes;
        int base = merge * n_nodes + node;
        if (src_counts[base] > dst_counts[base]) dst_counts[base] = src_counts[base];
        if (src_violations[base] > dst_violations[base]) dst_violations[base] = src_violations[base];
    }
}

__global__ void bloom_merge_kernel(
    uint64_t * __restrict__ dst, const uint64_t * __restrict__ src,
    int n_words, int n_merges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_words * n_merges) {
        dst[idx] |= src[idx];
    }
}

__global__ void sketch_merge_kernel(
    uint64_t * __restrict__ dst, const uint64_t * __restrict__ src,
    int total_cells, int n_merges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_cells * n_merges) {
        if (src[idx] > dst[idx]) dst[idx] = src[idx];
    }
}

// Batch merge: merge N pairs of full states
__global__ void state_merge_batch(
    uint64_t * __restrict__ dst_counts, uint64_t * __restrict__ src_counts,
    uint8_t * __restrict__ dst_active, const uint8_t * __restrict__ src_active,
    int n_nodes, int n_constraints, int n_pairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair = idx / max(n_nodes, n_constraints);
    int elem = idx % max(n_nodes, n_constraints);

    if (pair < n_pairs) {
        // G-Counter part
        if (elem < n_nodes) {
            int base_d = pair * n_nodes + elem;
            int base_s = pair * n_nodes + elem;
            if (src_counts[base_s] > dst_counts[base_d])
                dst_counts[base_d] = src_counts[base_s];
        }
        // OR-Set part
        if (elem < n_constraints) {
            int base_d = pair * n_constraints + elem;
            int base_s = pair * n_constraints + elem;
            dst_active[base_d] = dst_active[base_s] | dst_active[base_d];
        }
    }
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(void) {
    printf("=== CRDT Merge Benchmark — CUDA (RTX 4050, sm_86) ===\n\n");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s, SM %d.%d, %d SMs, %.0f MHz\n\n",
        prop.name, prop.major, prop.minor, prop.multiProcessorCount,
        prop.clockRate / 1000.0);

    int n_merges = 10000;
    int threads = 256;

    // ---- G-Counter benchmark ----
    {
        size_t bytes = n_merges * N_NODES * sizeof(uint64_t);
        uint64_t *d_dst_c, *d_src_c, *d_dst_v, *d_src_v;
        CUDA_CHECK(cudaMalloc(&d_dst_c, bytes));
        CUDA_CHECK(cudaMalloc(&d_src_c, bytes));
        CUDA_CHECK(cudaMalloc(&d_dst_v, bytes));
        CUDA_CHECK(cudaMalloc(&d_src_v, bytes));

        // Init
        uint64_t *h_c = (uint64_t*)calloc(n_merges * N_NODES, sizeof(uint64_t));
        for (int i = 0; i < n_merges * N_NODES; i++) h_c[i] = rand() % 1000;
        CUDA_CHECK(cudaMemcpy(d_dst_c, h_c, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_src_c, h_c, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dst_v, h_c, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_src_v, h_c, bytes, cudaMemcpyHostToDevice));
        free(h_c);

        int total = n_merges * N_NODES;
        int blocks = (total + threads - 1) / threads;

        // Warmup
        gcounter_merge_kernel<<<blocks, threads>>>(d_dst_c, d_dst_v, d_src_c, d_src_v, N_NODES, n_merges);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t start = now_ns();
        for (int rep = 0; rep < 100; rep++) {
            gcounter_merge_kernel<<<blocks, threads>>>(d_dst_c, d_dst_v, d_src_c, d_src_v, N_NODES, n_merges);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t elapsed = now_ns() - start;

        double ns_per_merge = (double)elapsed / (100.0 * n_merges);
        double merges_per_sec = 1e9 / ns_per_merge;
        printf("  G-Counter batch merge (%d states, %d nodes):\n", n_merges, N_NODES);
        printf("    %.1f ns/merge, %.0f merges/s\n", ns_per_merge, merges_per_sec);
        printf("    Total: %.0fM element merges/s\n\n", merges_per_sec * N_NODES / 1e6);

        CUDA_CHECK(cudaFree(d_dst_c));
        CUDA_CHECK(cudaFree(d_src_c));
        CUDA_CHECK(cudaFree(d_dst_v));
        CUDA_CHECK(cudaFree(d_src_v));
    }

    // ---- Bloom filter benchmark ----
    {
        size_t bytes = n_merges * BLOOM_WORDS * sizeof(uint64_t);
        uint64_t *d_dst, *d_src;
        CUDA_CHECK(cudaMalloc(&d_dst, bytes));
        CUDA_CHECK(cudaMalloc(&d_src, bytes));

        uint64_t *h = (uint64_t*)calloc(n_merges * BLOOM_WORDS, sizeof(uint64_t));
        for (int i = 0; i < n_merges * BLOOM_WORDS; i++) h[i] = rand();
        CUDA_CHECK(cudaMemcpy(d_dst, h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_src, h, bytes, cudaMemcpyHostToDevice));
        free(h);

        int total = n_merges * BLOOM_WORDS;
        int blocks = (total + threads - 1) / threads;

        bloom_merge_kernel<<<blocks, threads>>>(d_dst, d_src, BLOOM_WORDS, n_merges);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t start = now_ns();
        for (int rep = 0; rep < 100; rep++) {
            bloom_merge_kernel<<<blocks, threads>>>(d_dst, d_src, BLOOM_WORDS, n_merges);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t elapsed = now_ns() - start;

        double ns_per_merge = (double)elapsed / (100.0 * n_merges);
        printf("  Bloom filter batch merge (%d states, %d words):\n", n_merges, BLOOM_WORDS);
        printf("    %.1f ns/merge, %.0f merges/s\n", ns_per_merge, 1e9 / ns_per_merge);
        printf("    %.1f GB/s throughput\n\n",
            (2.0 * bytes * 100.0 / (elapsed / 1e9)) / 1e9);

        CUDA_CHECK(cudaFree(d_dst));
        CUDA_CHECK(cudaFree(d_src));
    }

    // ---- Count-Min Sketch benchmark ----
    {
        int total_cells = SKETCH_D * SKETCH_W;
        size_t bytes = n_merges * total_cells * sizeof(uint64_t);
        uint64_t *d_dst, *d_src;
        CUDA_CHECK(cudaMalloc(&d_dst, bytes));
        CUDA_CHECK(cudaMalloc(&d_src, bytes));

        uint64_t *h = (uint64_t*)calloc(n_merges * total_cells, sizeof(uint64_t));
        for (int i = 0; i < n_merges * total_cells; i++) h[i] = rand() % 10000;
        CUDA_CHECK(cudaMemcpy(d_dst, h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_src, h, bytes, cudaMemcpyHostToDevice));
        free(h);

        int total = n_merges * total_cells;
        int blocks = (total + threads - 1) / threads;

        sketch_merge_kernel<<<blocks, threads>>>(d_dst, d_src, total_cells, n_merges);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t start = now_ns();
        for (int rep = 0; rep < 100; rep++) {
            sketch_merge_kernel<<<blocks, threads>>>(d_dst, d_src, total_cells, n_merges);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t elapsed = now_ns() - start;

        double ns_per_merge = (double)elapsed / (100.0 * n_merges);
        printf("  Sketch batch merge (%d states, %d×%d cells):\n", n_merges, SKETCH_D, SKETCH_W);
        printf("    %.1f ns/merge, %.0f merges/s\n", ns_per_merge, 1e9 / ns_per_merge);
        printf("    %.1f GB/s throughput\n\n",
            (2.0 * bytes * 100.0 / (elapsed / 1e9)) / 1e9);

        CUDA_CHECK(cudaFree(d_dst));
        CUDA_CHECK(cudaFree(d_src));
    }

    return 0;
}
