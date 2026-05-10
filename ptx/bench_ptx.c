// CRDT Merge — PTX (hand-written assembly)
// Direct PTX for G-Counter element-wise max and Bloom filter bitwise OR
// This is as close to the metal as we can get

#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define N_NODES 32
#define BLOOM_WORDS 94
#define N_RUNS 1000000

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// ---- Inline PTX for single G-Counter element merge ----
// Uses PTX max.u64 instruction directly
static inline void ptx_gcounter_merge_element(uint64_t *dst, const uint64_t *src) {
    // PTX: ld.global.u64 + max.u64 + st.global.u64
    asm volatile (
        "{\n\t"
        ".reg .u64 d, s, m;\n\t"
        "ld.global.u64 s, [%1];\n\t"
        "ld.global.u64 d, [%0];\n\t"
        "max.u64 m, d, s;\n\t"
        "st.global.u64 [%0], m;\n\t"
        "}\n\t"
        : 
        : "l"(dst), "l"(src)
        : "memory"
    );
}

// ---- Inline PTX for Bloom filter OR ----
static inline void ptx_bloom_or_element(uint64_t *dst, const uint64_t *src) {
    asm volatile (
        "{\n\t"
        ".reg .u64 d, s, r;\n\t"
        "ld.global.u64 s, [%1];\n\t"
        "ld.global.u64 d, [%0];\n\t"
        "or.b64 r, d, s;\n\t"
        "st.global.u64 [%0], r;\n\t"
        "}\n\t"
        : 
        : "l"(dst), "l"(src)
        : "memory"
    );
}

// ---- C equivalents for comparison ----
static inline void c_max_merge(uint64_t *dst, const uint64_t *src, int n) {
    for (int i = 0; i < n; i++)
        if (src[i] > dst[i]) dst[i] = src[i];
}

static inline void c_bloom_merge(uint64_t *dst, const uint64_t *src, int n) {
    for (int i = 0; i < n; i++)
        dst[i] |= src[i];
}

int main(void) {
    printf("=== CRDT Merge — PTX Inline Assembly vs C ===\n");
    printf("Runs: %d\n\n", N_RUNS);

    // ---- G-Counter: PTX max.u64 vs C ternary ----
    {
        uint64_t dst[N_NODES], src[N_NODES];
        for (int i = 0; i < N_NODES; i++) { dst[i] = rand(); src[i] = rand(); }

        // C baseline
        uint64_t start = now_ns();
        for (int r = 0; r < N_RUNS; r++) {
            c_max_merge(dst, src, N_NODES);
        }
        uint64_t c_elapsed = now_ns() - start;

        // PTX inline
        start = now_ns();
        for (int r = 0; r < N_RUNS; r++) {
            for (int i = 0; i < N_NODES; i++) {
                ptx_gcounter_merge_element(&dst[i], &src[i]);
            }
        }
        uint64_t ptx_elapsed = now_ns() - start;

        printf("  G-Counter merge (%d elements):\n", N_NODES);
        printf("    C (ternary):    %8.1f ns/op\n", (double)c_elapsed / N_RUNS);
        printf("    PTX (max.u64):  %8.1f ns/op\n", (double)ptx_elapsed / N_RUNS);
        printf("    Speedup: %.2fx\n\n", (double)c_elapsed / ptx_elapsed);
    }

    // ---- Bloom filter: PTX or.b64 vs C |= ----
    {
        uint64_t dst[BLOOM_WORDS], src[BLOOM_WORDS];
        for (int i = 0; i < BLOOM_WORDS; i++) { dst[i] = rand(); src[i] = rand(); }

        // C baseline
        uint64_t start = now_ns();
        for (int r = 0; r < N_RUNS; r++) {
            c_bloom_merge(dst, src, BLOOM_WORDS);
        }
        uint64_t c_elapsed = now_ns() - start;

        // PTX inline
        start = now_ns();
        for (int r = 0; r < N_RUNS; r++) {
            for (int i = 0; i < BLOOM_WORDS; i++) {
                ptx_bloom_or_element(&dst[i], &src[i]);
            }
        }
        uint64_t ptx_elapsed = now_ns() - start;

        printf("  Bloom filter merge (%d words):\n", BLOOM_WORDS);
        printf("    C (|=):         %8.1f ns/op\n", (double)c_elapsed / N_RUNS);
        printf("    PTX (or.b64):   %8.1f ns/op\n", (double)ptx_elapsed / N_RUNS);
        printf("    Speedup: %.2fx\n\n", (double)c_elapsed / ptx_elapsed);
    }

    // ---- Vectorized C (loop unrolled x4) ----
    {
        uint64_t dst[BLOOM_WORDS * 4], src[BLOOM_WORDS * 4]; // 4x for fair comparison
        for (int i = 0; i < BLOOM_WORDS * 4; i++) { dst[i] = rand(); src[i] = rand(); }

        uint64_t start = now_ns();
        for (int r = 0; r < N_RUNS; r++) {
            int n = BLOOM_WORDS * 4;
            int i = 0;
            for (; i + 3 < n; i += 4) {
                dst[i] |= src[i];
                dst[i+1] |= src[i+1];
                dst[i+2] |= src[i+2];
                dst[i+3] |= src[i+3];
            }
            for (; i < n; i++) dst[i] |= src[i];
        }
        uint64_t elapsed = now_ns() - start;
        printf("  Bloom merge (unrolled ×4, %d words):\n", BLOOM_WORDS * 4);
        printf("    %.1f ns/op, %.0fM words/s\n\n",
            (double)elapsed / N_RUNS,
            BLOOM_WORDS * 4.0 / ((double)elapsed / N_RUNS / 1000.0));
    }

    return 0;
}
