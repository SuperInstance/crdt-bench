# CRDT Merge Cross-Language Bake-Off

Performance comparison of CRDT merge operations across C, Rust, CUDA, and PTX.
Same algorithms, same hardware, different languages.

## Hardware
- **CPU**: AMD Ryzen AI 9 HX 370 (Zen 5, 12C/24T, AVX-512)
- **GPU**: NVIDIA RTX 4050 Laptop (Ada, sm_89, 20 SMs)
- **RAM**: 16GB DDR5

## Results

### CPU: G-Counter Merge (32 elements, element-wise max)

| Language | Implementation | Time | Throughput |
|----------|---------------|------|------------|
| Rust | for loop + black_box | 17.5 ns | 57.3M ops/s |
| Rust | unsafe ptr | 18.5 ns | 54.0M ops/s |
| C | gcc -O2 -march=native | 10.5 ns | 95.3M ops/s |

### CPU: Bloom Filter Merge (94 words, bitwise OR)

| Language | Implementation | Time | Throughput |
|----------|---------------|------|------------|
| Rust | unrolled ×4 | 31.8 ns | 31.5M ops/s |
| Rust | for loop | 51.9 ns | 19.3M ops/s |
| C | gcc -O2 -march=native | 25.6 ns | 39.1M ops/s |

### CPU: Full State Merge (64 active constraints)

| Language | Time | Throughput |
|----------|------|------------|
| C | 149.9 ns | 6.7M ops/s |

### CPU: Count-Min Sketch Merge (7×1000 cells)

| Language | Time | Throughput |
|----------|------|------------|
| Rust | safe loop | 839 ns | 1.2M ops/s |
| C | 2448 ns | 409K ops/s |

### GPU: Batch Merge (10,000 states in parallel)

| Operation | Time/merge | Throughput | Bandwidth |
|-----------|-----------|------------|-----------|
| Bloom filter OR | 4.4 ns | 227.5M merges/s | 342 GB/s |
| G-Counter max | 5.5 ns | 180.4M merges/s | — |
| Sketch max (7K cells) | 1620 ns | 617K merges/s | 69 GB/s |

## GPU SASS Analysis (RTX 4050, actual machine code)

The Bloom filter OR merge compiles to **3 instructions per element**:
```
LDG.E.64       R2, [R2.64]        // Load src
LDG.E.64       R6, [R4.64]        // Load dst
LOP3.LUT       R6, R6, R2, 0xfc  // OR (lookup table 0xfc = bitwise OR)
STG.E.64       [R4.64], R6        // Store result
```

The G-Counter max merge compiles to **5 instructions per element**:
```
LDG.E.64       // Load src
LDG.E.64       // Load dst
ISETP          // Compare
SEL            // Select max
STG.E.64       // Store result
```

**Bloom merge is 40% fewer GPU instructions than G-Counter max.**

## Key Findings

1. **Bloom CRDT wins on GPU**: OR is 3 instructions vs 5 for max → 227M vs 180M merges/s
2. **C wins on CPU**: gcc vectorizes the loop better than rustc (95M vs 57M ops/s for G-Counter)
3. **GPU destroys CPU on batch work**: 227M merges/s (GPU) vs 39M (C CPU) = **5.8x**
4. **Sketch merge is memory-bound**: 56KB data structure → cache misses dominate
5. **PTX confirms**: `LOP3.LUT` with `0xfc` is the OR operation — Turing/Ada's triple-LOP saves a register

## Mojo (not yet installed)

Mojo's `SIMD[DType.uint64, N]` would give vectorized element-wise operations
in a single instruction. For N_NODES=32: two 256-bit vectors, one max each.
Expected: comparable to C, potentially faster with explicit SIMD.

## Build & Run

```bash
# C
cd c && gcc -O2 -march=native -o bench bench.c -lrt && ./bench

# Rust
cd rust && rustc -O -C target-cpu=native -o bench_real bench_real.rs && ./bench_real

# CUDA
cd cuda && nvcc -O2 -arch=sm_86 -o bench bench.cu && ./bench

# PTX inspection
cd cuda && nvcc -O2 -arch=sm_86 -ptx -o bench.ptx bench.cu
cuobjdump -sass bench | grep LOP3
```

## License

MIT
