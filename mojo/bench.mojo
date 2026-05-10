# CRDT Merge Micro-Benchmark — Mojo
# Mojo combines Python syntax with systems-level performance
# This will run when Mojo is installed

from benchmark import benchmark
from math import max

alias N_NODES = 32
alias BLOOM_WORDS = 94
alias N_RUNS = 1_000_000

@value
struct GCounter:
    var counts: SIMD[DType.uint64, N_NODES]
    var violations: SIMD[DType.uint64, N_NODES]

    fn __init__(inout self):
        self.counts = SIMD[DType.uint64, N_NODES](0)
        self.violations = SIMD[DType.uint64, N_NODES](0)

    fn merge(inout self, other: Self):
        # SIMD element-wise max — Mojo's bread and butter
        self.counts = max(self.counts, other.counts)
        self.violations = max(self.violations, other.violations)

@value
struct BloomFilter:
    var bits: SIMD[DType.uint64, BLOOM_WORDS]

    fn __init__(inout self):
        self.bits = SIMD[DType.uint64, BLOOM_WORDS](0)

    fn merge(inout self, other: Self):
        # SIMD bitwise OR — single instruction for entire filter
        self.bits = self.bits | other.bits

fn main():
    print("=== CRDT Merge Benchmark — Mojo ===")
    print("Runs:", N_RUNS)

    var gc_a = GCounter()
    var gc_b = GCounter()
    gc_a.counts = SIMD[DType.uint64, N_NODES](100)
    gc_b.counts = SIMD[DType.uint64, N_NODES](200)

    @benchmark(iterations=N_RUNS)
    fn bench_gcounter():
        gc_a.merge(gc_b)

    var bl_a = BloomFilter()
    var bl_b = BloomFilter()
    bl_a.bits = SIMD[DType.uint64, BLOOM_WORDS](0xFF)
    bl_b.bits = SIMD[DType.uint64, BLOOM_WORDS](0xAA)

    @benchmark(iterations=N_RUNS)
    fn bench_bloom():
        bl_a.merge(bl_b)

    print("\n  Mojo advantage: SIMD[DType.uint64, N] gives vectorized")
    print("  element-wise max/or in a SINGLE instruction for the whole array.")
    print("  For N_NODES=32: 2 × 256-bit vectors = 1 max instruction each.")
    print("  For BLOOM_WORDS=94: ~3 vector instructions for entire merge.")
