// CRDT Merge Micro-Benchmark — Pure C
// G-Counter merge: element-wise max over per-node counts
// OR-Set merge: union of constraint IDs with tombstone check
// Full state merge: composite of above

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define N_NODES 32
#define N_CONSTRAINTS 256
#define N_RUNS 1000000

typedef struct {
    uint64_t counts[N_NODES];
    uint64_t violations[N_NODES];
} gcounter_t;

typedef struct {
    uint8_t active[N_CONSTRAINTS];      // 1 = active, 0 = removed
    uint8_t tombstones[N_CONSTRAINTS];  // 1 = removed at some point
} orset_t;

typedef struct {
    gcounter_t counter;
    orset_t constraints;
    int32_t pos_a;
    int32_t pos_b;
    int64_t norm;
    uint64_t version;
} state_t;

// ---- G-Counter merge: max per element ----
static inline void gcounter_merge(gcounter_t *dst, const gcounter_t *src) {
    for (int i = 0; i < N_NODES; i++) {
        if (src->counts[i] > dst->counts[i]) dst->counts[i] = src->counts[i];
        if (src->violations[i] > dst->violations[i]) dst->violations[i] = src->violations[i];
    }
}

// ---- OR-Set merge: union with tombstone removal ----
static inline void orset_merge(orset_t *dst, const orset_t *src) {
    for (int i = 0; i < N_CONSTRAINTS; i++) {
        // Merge tombstones
        if (src->tombstones[i]) dst->tombstones[i] = 1;
        // Merge active (union)
        if (src->active[i]) dst->active[i] = 1;
        // Remove active if tombstoned
        if (dst->tombstones[i]) dst->active[i] = 0;
    }
}

// ---- Full state merge ----
static inline void state_merge(state_t *dst, const state_t *src) {
    gcounter_merge(&dst->counter, &src->counter);
    orset_merge(&dst->constraints, &src->constraints);
    // Position: lower norm wins
    if (src->norm < dst->norm) {
        dst->pos_a = src->pos_a;
        dst->pos_b = src->pos_b;
        dst->norm = src->norm;
    }
    dst->version = (dst->version > src->version) ? dst->version : src->version;
}

// ---- Eisenstein norm ----
static inline int64_t e12_norm(int32_t a, int32_t b) {
    int64_t la = a, lb = b;
    return la * la - la * lb + lb * lb;
}

// ---- Bloom filter merge: bitwise OR on 64-bit words ----
#define BLOOM_WORDS 94  // ~6000 bits for 10K items at 1% FPR
typedef struct {
    uint64_t bits[BLOOM_WORDS];
} bloom_t;

static inline void bloom_merge(bloom_t *dst, const bloom_t *src) {
    for (int i = 0; i < BLOOM_WORDS; i++) {
        dst->bits[i] |= src->bits[i];
    }
}

// ---- Count-Min Sketch merge: element-wise max ----
#define SKETCH_D 7
#define SKETCH_W 1000
typedef struct {
    uint64_t counters[SKETCH_D][SKETCH_W];
} sketch_t;

static inline void sketch_merge(sketch_t *dst, const sketch_t *src) {
    for (int d = 0; d < SKETCH_D; d++) {
        for (int w = 0; w < SKETCH_W; w++) {
            if (src->counters[d][w] > dst->counters[d][w])
                dst->counters[d][w] = src->counters[d][w];
        }
    }
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#define BENCH(name, merge_fn, init_fn, type) do { \
    type a, b; \
    init_fn(&a); init_fn(&b); \
    uint64_t start = now_ns(); \
    for (int i = 0; i < N_RUNS; i++) { \
        merge_fn(&a, &b); \
    } \
    uint64_t elapsed = now_ns() - start; \
    double ns_per = (double)elapsed / N_RUNS; \
    printf("  %-30s %8.1f ns/op  %10.0f ops/s\n", name, ns_per, 1e9 / ns_per); \
} while(0)

static void gc_init(gcounter_t *g) { memset(g, 0, sizeof(*g)); g->counts[0] = 100; g->counts[5] = 200; }
static void ors_init(orset_t *o) { memset(o, 0, sizeof(*o)); for(int i=0;i<128;i++) o->active[i]=1; }
static void st_init(state_t *s) { memset(s, 0, sizeof(*s)); s->counter.counts[0]=100; s->norm=e12_norm(3,0); for(int i=0;i<64;i++) s->constraints.active[i]=1; }
static void bl_init(bloom_t *b) { memset(b, 0, sizeof(*b)); b->bits[0]=0xFF; b->bits[10]=0xAA; }
static void sk_init(sketch_t *s) { memset(s, 0, sizeof(*s)); s->counters[0][0]=100; s->counters[3][500]=200; }

int main(void) {
    printf("=== CRDT Merge Benchmark — C (gcc -O2) ===\n");
    printf("Runs: %d, Nodes: %d, Constraints: %d\n\n", N_RUNS, N_NODES, N_CONSTRAINTS);

    BENCH("G-Counter merge (32 nodes)", gcounter_merge, gc_init, gcounter_t);
    BENCH("OR-Set merge (256 constraints)", orset_merge, ors_init, orset_t);
    BENCH("Full state merge (64 active)", state_merge, st_init, state_t);
    BENCH("Bloom filter merge (6008 bits)", bloom_merge, bl_init, bloom_t);
    BENCH("Sketch merge (7×1000)", sketch_merge, sk_init, sketch_t);

    printf("\n--- Data structure sizes ---\n");
    printf("  gcounter_t: %zu bytes\n", sizeof(gcounter_t));
    printf("  orset_t:    %zu bytes\n", sizeof(orset_t));
    printf("  state_t:    %zu bytes\n", sizeof(state_t));
    printf("  bloom_t:    %zu bytes\n", sizeof(bloom_t));
    printf("  sketch_t:   %zu bytes\n", sizeof(sketch_t));

    return 0;
}
