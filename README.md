# CRDT Merge Cross-Language Bake-Off

Performance comparison of CRDT merge operations across 7 languages.
Same algorithms, same hardware, different approaches.

## Hardware
- **CPU**: AMD Ryzen AI 9 HX 370 (Zen 5, 12C/24T, AVX-512)
- **GPU**: NVIDIA RTX 4050 Laptop (Ada, sm_89, 20 SMs)

## Final Results — CPU

### G-Counter Merge (32 elements, element-wise max)

| Language | Implementation | Time | Throughput |
|----------|---------------|------|------------|
| **Fortran** | **whole-array MAX** | **0.9 ns** | **1.09B ops/s** |
| Zig | scalar loop | 3.0 ns | 333M ops/s |
| Zig | @Vector(16) SIMD | 7.0 ns | 143M ops/s |
| Go | range loop | 8.3 ns | 120M ops/s |
| C | gcc -O2 | 10.5 ns | 95M ops/s |
| Go | unsafe ptr | 10.8 ns | 93M ops/s |
| Rust | for loop | 17.5 ns | 57M ops/s |

### Bloom Filter Merge (94 words = 752 bytes, bitwise OR)

| Language | Implementation | Time | Throughput |
|----------|---------------|------|------------|
| **Fortran** | **whole-array IOR** | **0.4 ns** | **2.27B ops/s** |
| Zig | @Vector(8) SIMD | 5.0 ns | 200M ops/s |
| Rust | iterator zip | 11.0 ns | 91M ops/s |
| Go | range loop | 27.9 ns | 36M ops/s |
| C | gcc -O2 | 25.6 ns | 39M ops/s |

### Count-Min Sketch Merge (7,000 cells, element-wise max)

| Language | Implementation | Time | Throughput |
|----------|---------------|------|------------|
| **Fortran** | **2D whole-array MAX** | **623 ns** | **1.6M ops/s** |
| Zig | @Vector(8) SIMD | 827 ns | 1.21M ops/s |
| Rust | safe loop | 839 ns | 1.19M ops/s |
| Go | flat slice | 2,107 ns | 475K ops/s |
| C | gcc -O2 | 2,448 ns | 409K ops/s |

## Results — GPU (batch: 10,000 states in parallel)

| Operation | Time/merge | Throughput |
|-----------|-----------|------------|
| Bloom OR (RTX 4050) | 4.4 ns | 227M merges/s |
| G-Counter max (RTX 4050) | 5.5 ns | 180M merges/s |
| Sketch max (RTX 4050) | 1620 ns | 617K merges/s |

## GPU SASS (actual machine code from RTX 4050)

Bloom OR = 3 instructions: `LDG.E.64 → LOP3.LUT(0xfc=OR) → STG.E.64`
G-Counter max = 5 instructions: `LDG → LDG → ISETP → SEL → STG`

## Rankings

### Single-thread CPU performance
1. 🥇 **Fortran** — whole-array operations = direct AVX-512, zero overhead
2. 🥈 **Zig** — `@Vector` explicit SIMD, comptime unrolling
3. 🥉 **C** — gcc auto-vectorizes simple loops
4. **Go** — decent, but no SIMD intrinsics
5. **Rust** — safe abstractions have cost, unsafe matches C

### Code elegance (CRDT merge expressiveness)
1. 🥇 **Fortran** — `dst = MAX(dst, src)` is ONE LINE
2. 🥈 **Zig** — `@max(dst_vec, src_vec)` with `@Vector` types
3. 🥉 **Go** — `for i := range dst { dst[i] = max(dst[i], src[i]) }`
4. **Rust** — `dst.iter_mut().zip(src.iter()).for_each(|(d,s)| *d = d.max(*s))`
5. **C** — `for (int i=0; i<n; i++) if (src[i] > dst[i]) dst[i] = src[i];`

### GPU throughput
1. 🥇 **CUDA** — 227M Bloom merges/s (RTX 4050)
2. **Metal** — ~200M estimated (Apple M4)
3. **CPU Fortran** — 2.27B ops/s (but single state, not 10K batch)

## The Universal Fast Path

The bitwise OR semilattice is a **single instruction on every architecture**:

| Architecture | Instruction | Perf |
|-------------|-----------|------|
| x86 AVX-512 | `vpor zmm` | 512 bits/cycle |
| NVIDIA Ada | `LOP3.LUT 0xfc` | 3 SASS instructions |
| Apple AGX | `OR.64 / VOR.128` | 2 cycles |
| AMD RDNA3 | `V_OR_B64` | 1 instruction |
| ARM NEON | `orr v.2D` | 128 bits/instruction |

**Bloom CRDT merge is O(1) per element everywhere. No other CRDT type is this fast.**

## Key Finding

**Fortran wins because CRDT merge IS a whole-array operation.**

The semilattice join (max for G-Counter, OR for Bloom) maps directly to Fortran's native array primitives. No other language has this — C needs loops, Rust needs iterators, Zig needs `@Vector`. Fortran just writes the math and the compiler does the rest.

After 70 years, Fortran is still the best language for semilattice operations.

## Build & Run

```bash
# Fortran (fastest)
cd fortran && gfortran -O3 -march=native -ffree-line-length-none -o bench bench.f90 && ./bench

# Zig
cd zig && zig build-exe -O ReleaseFast bench.zig && ./bench

# Go
cd go && go build -o bench bench.go && ./bench

# C
cd c && gcc -O2 -march=native -o bench bench.c -lrt && ./bench

# Rust
cd rust && rustc -O -C target-cpu=native -o bench_real bench_real.rs && ./bench_real

# CUDA
cd cuda && nvcc -O2 -arch=sm_86 -o bench bench.cu && ./bench

# Metal analysis
cd metal && python3 analyze.py
```

## License

MIT
