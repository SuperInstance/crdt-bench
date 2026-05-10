// CRDT Merge Micro-Benchmark — Rust
// Same operations as C version for direct comparison

use std::time::Instant;

const N_NODES: usize = 32;
const N_CONSTRAINTS: usize = 256;
const BLOOM_WORDS: usize = 94;
const SKETCH_D: usize = 7;
const SKETCH_W: usize = 1000;
const N_RUNS: usize = 1_000_000;

struct GCounter {
    counts: [u64; N_NODES],
    violations: [u64; N_NODES],
}

struct ORSet {
    active: [u8; N_CONSTRAINTS],
    tombstones: [u8; N_CONSTRAINTS],
}

struct BloomFilter {
    bits: [u64; BLOOM_WORDS],
}

struct Sketch {
    counters: [[u64; SKETCH_W]; SKETCH_D],
}

fn gcounter_merge(dst: &mut GCounter, src: &GCounter) {
    for i in 0..N_NODES {
        dst.counts[i] = dst.counts[i].max(src.counts[i]);
        dst.violations[i] = dst.violations[i].max(src.violations[i]);
    }
}

fn orset_merge(dst: &mut ORSet, src: &ORSet) {
    for i in 0..N_CONSTRAINTS {
        if src.tombstones[i] != 0 { dst.tombstones[i] = 1; }
        if src.active[i] != 0 { dst.active[i] = 1; }
        if dst.tombstones[i] != 0 { dst.active[i] = 0; }
    }
}

fn bloom_merge(dst: &mut BloomFilter, src: &BloomFilter) {
    for i in 0..BLOOM_WORDS {
        dst.bits[i] |= src.bits[i];
    }
}

fn sketch_merge(dst: &mut Sketch, src: &Sketch) {
    for d in 0..SKETCH_D {
        for w in 0..SKETCH_W {
            dst.counters[d][w] = dst.counters[d][w].max(src.counters[d][w]);
        }
    }
}

fn bench<F: FnMut()>(name: &str, mut f: F) {
    // Warmup
    for _ in 0..1000 { f(); }
    
    let start = Instant::now();
    for _ in 0..N_RUNS { f(); }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / N_RUNS as f64;
    println!("  {:<35} {:>8.1} ns/op  {:>10.0} ops/s", name, ns_per, 1e9 / ns_per);
}

fn main() {
    println!("=== CRDT Merge Benchmark — Rust (rustc -O2, Zen 5) ===");
    println!("Runs: {}, Nodes: {}, Constraints: {}\n", N_RUNS, N_NODES, N_CONSTRAINTS);

    let mut gc_a = GCounter { counts: [0; N_NODES], violations: [0; N_NODES] };
    let mut gc_b = GCounter { counts: [0; N_NODES], violations: [0; N_NODES] };
    gc_a.counts[0] = 100; gc_b.counts[5] = 200;

    let mut ors_a = ORSet { active: [0; N_CONSTRAINTS], tombstones: [0; N_CONSTRAINTS] };
    let mut ors_b = ORSet { active: [0; N_CONSTRAINTS], tombstones: [0; N_CONSTRAINTS] };
    for i in 0..128 { ors_a.active[i] = 1; }

    let mut bl_a = BloomFilter { bits: [0; BLOOM_WORDS] };
    let mut bl_b = BloomFilter { bits: [0; BLOOM_WORDS] };
    bl_a.bits[0] = 0xFF; bl_b.bits[10] = 0xAA;

    let mut sk_a = Sketch { counters: [[0u64; SKETCH_W]; SKETCH_D] };
    let mut sk_b = Sketch { counters: [[0u64; SKETCH_W]; SKETCH_D] };
    sk_a.counters[0][0] = 100; sk_b.counters[3][500] = 200;

    bench("G-Counter merge (32 nodes)", || { gcounter_merge(&mut gc_a, &gc_b); });
    bench("OR-Set merge (256 constraints)", || { orset_merge(&mut ors_a, &ors_b); });
    bench("Bloom filter merge (752 bytes)", || { bloom_merge(&mut bl_a, &bl_b); });
    bench("Sketch merge (7×1000)", || { sketch_merge(&mut sk_a, &sk_b); });

    // SIMD-style unrolled Bloom merge
    bench("Bloom merge (unrolled ×4)", || {
        let src = &bl_b.bits;
        let dst = &mut bl_a.bits;
        let mut i = 0;
        while i + 3 < BLOOM_WORDS {
            dst[i] |= src[i]; dst[i+1] |= src[i+1];
            dst[i+2] |= src[i+2]; dst[i+3] |= src[i+3];
            i += 4;
        }
        while i < BLOOM_WORDS { dst[i] |= src[i]; i += 1; }
    });

    // Iterator-based Bloom merge
    bench("Bloom merge (iterator zip)", || {
        bl_a.bits.iter_mut().zip(bl_b.bits.iter()).for_each(|(d, s)| *d |= *s);
    });

    // Unsafe raw pointer Bloom merge
    bench("Bloom merge (unsafe ptr)", || {
        unsafe {
            let dst = bl_a.bits.as_mut_ptr();
            let src = bl_b.bits.as_ptr();
            for i in 0..BLOOM_WORDS {
                *dst.add(i) |= *src.add(i);
            }
        }
    });

    println!("\n--- Data structure sizes ---");
    println!("  GCounter:    {} bytes", std::mem::size_of::<GCounter>());
    println!("  ORSet:       {} bytes", std::mem::size_of::<ORSet>());
    println!("  BloomFilter: {} bytes", std::mem::size_of::<BloomFilter>());
    println!("  Sketch:      {} bytes", std::mem::size_of::<Sketch>());
}
