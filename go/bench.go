package main

import (
	"fmt"
	"time"
	"unsafe"
)

const (
	N_NODES      = 32
	BLOOM_WORDS  = 94
	SKETCH_D     = 7
	SKETCH_W     = 1000
	N_RUNS       = 2_000_000
)

// ---- G-Counter merge: element-wise max ----
func gcounterMergeScalar(dst, src []uint64) {
	for i := range dst {
		if src[i] > dst[i] {
			dst[i] = src[i]
		}
	}
}

func gcounterMergeUnsafe(dst, src []uint64) {
	d := unsafe.Slice((*uint64)(unsafe.Pointer(&dst[0])), len(dst))
	s := unsafe.Slice((*uint64)(unsafe.Pointer(&src[0])), len(src))
	for i := range d {
		if s[i] > d[i] {
			d[i] = s[i]
		}
	}
}

// ---- Bloom filter merge: bitwise OR ----
func bloomMergeScalar(dst, src []uint64) {
	for i := range dst {
		dst[i] |= src[i]
	}
}

func bloomMergeUnrolled4(dst, src []uint64) {
	n := len(dst)
	i := 0
	for i+3 < n {
		dst[i] |= src[i]
		dst[i+1] |= src[i+1]
		dst[i+2] |= src[i+2]
		dst[i+3] |= src[i+3]
		i += 4
	}
	for i < n {
		dst[i] |= src[i]
		i++
	}
}

func bloomMergeUnsafe(dst, src []uint64) {
	d := unsafe.Slice((*uint64)(unsafe.Pointer(&dst[0])), len(dst))
	s := unsafe.Slice((*uint64)(unsafe.Pointer(&src[0])), len(src))
	for i := range d {
		d[i] |= s[i]
	}
}

// ---- Count-Min Sketch merge: element-wise max ----
func sketchMergeScalar(dst, src [][SKETCH_W]uint64) {
	for d := range dst {
		for w := 0; w < SKETCH_W; w++ {
			if src[d][w] > dst[d][w] {
				dst[d][w] = src[d][w]
			}
		}
	}
}

func sketchMergeFlat(dst, src []uint64) {
	for i := range dst {
		if src[i] > dst[i] {
			dst[i] = src[i]
		}
	}
}

// ---- Eisenstein norm ----
func e12Norm(a, b int32) int64 {
	la, lb := int64(a), int64(b)
	return la*la - la*lb + lb*lb
}

func bench(name string, f func()) {
	// Warmup
	for i := 0; i < 1000; i++ {
		f()
	}
	start := time.Now()
	for i := 0; i < N_RUNS; i++ {
		f()
	}
	elapsed := time.Since(start)
	nsPer := float64(elapsed.Nanoseconds()) / float64(N_RUNS)
	opsPerS := 1e9 / nsPer
	fmt.Printf("  %-45s %8.1f ns  %12.0f ops/s\n", name, nsPer, opsPerS)
}

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  CRDT MERGE — Go 1.24 (Zen 5)                               ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════╝")
	fmt.Printf("  Runs: %d, Nodes: %d, Bloom: %d words\n\n", N_RUNS, N_NODES, BLOOM_WORDS)

	// G-Counter
	gcA := make([]uint64, N_NODES)
	gcB := make([]uint64, N_NODES)
	for i := range gcA {
		gcA[i] = uint64(i * i)
		gcB[i] = 100
	}

	bench("G-Counter: range loop max", func() { gcounterMergeScalar(gcA, gcB) })
	bench("G-Counter: unsafe ptr max", func() { gcounterMergeUnsafe(gcA, gcB) })

	fmt.Println()

	// Bloom
	blA := make([]uint64, BLOOM_WORDS)
	blB := make([]uint64, BLOOM_WORDS)
	for i := range blA {
		blA[i] = uint64(i * i * i)
		blB[i] = 0xAAAAAAAAAAAAAAAA
	}

	bench("Bloom: range loop |=", func() { bloomMergeScalar(blA, blB) })
	bench("Bloom: unrolled ×4 |=", func() { bloomMergeUnrolled4(blA, blB) })
	bench("Bloom: unsafe ptr |=", func() { bloomMergeUnsafe(blA, blB) })

	fmt.Println()

	// Sketch
	var skA [SKETCH_D][SKETCH_W]uint64
	var skB [SKETCH_D][SKETCH_W]uint64
	for d := 0; d < SKETCH_D; d++ {
		for w := 0; w < SKETCH_W; w++ {
			skA[d][w] = uint64((d + 1) * (w + 1))
			skB[d][w] = uint64((d + 2) * (w + 3))
		}
	}

	bench("Sketch: 2D loop max (7K cells)", func() { sketchMergeScalar(skA[:], skB[:]) })

	// Flat slice version
	flatA := make([]uint64, SKETCH_D*SKETCH_W)
	flatB := make([]uint64, SKETCH_D*SKETCH_W)
	for i := range flatA {
		flatA[i] = uint64(i + 1)
		flatB[i] = uint64(i + 3)
	}
	bench("Sketch: flat slice max (7K cells)", func() { sketchMergeFlat(flatA, flatB) })

	fmt.Println()
	fmt.Println("--- Go analysis ---")
	fmt.Println("  Go has no SIMD intrinsics (no @Vector like Zig)")
	fmt.Println("  unsafe.Slice bypasses bounds checking but doesn't add vectorization")
	fmt.Println("  GC pauses not visible in micro-benchmarks (<1µs per merge)")
	fmt.Println("  Go's strength: goroutine-based parallel merge, not single-thread speed")

	// Parallel merge test
	fmt.Println()
	fmt.Println("--- Parallel merge (4 goroutines, 10K states) ---")
	type State struct {
		counts []uint64
		bloom  []uint64
	}
	states := make([]State, 10000)
	for i := range states {
		states[i].counts = make([]uint64, N_NODES)
		states[i].bloom = make([]uint64, BLOOM_WORDS)
		for j := range states[i].counts {
			states[i].counts[j] = uint64(i + j)
		}
		for j := range states[i].bloom {
			states[i].bloom[j] = 0xAA
		}
	}
	src := State{
		counts: make([]uint64, N_NODES),
		bloom:  make([]uint64, BLOOM_WORDS),
	}
	for i := range src.counts { src.counts[i] = 999 }
	for i := range src.bloom { src.bloom[i] = 0xFF }

	// Sequential
	start := time.Now()
	for i := range states {
		gcounterMergeScalar(states[i].counts, src.counts)
		bloomMergeScalar(states[i].bloom, src.bloom)
	}
	seqMs := time.Since(start).Milliseconds()

	// Parallel (4 goroutines)
	done := make(chan bool, 4)
	chunk := len(states) / 4
	start = time.Now()
	for g := 0; g < 4; g++ {
		go func(from, to int) {
			for i := from; i < to; i++ {
				gcounterMergeScalar(states[i].counts, src.counts)
				bloomMergeScalar(states[i].bloom, src.bloom)
			}
			done <- true
		}(g*chunk, (g+1)*chunk)
	}
	for g := 0; g < 4; g++ { <-done }
	parMs := time.Since(start).Milliseconds()

	fmt.Printf("  Sequential: %d ms, Parallel (4 goroutines): %d ms, Speedup: %.1fx\n",
		seqMs, parMs, float64(seqMs)/float64(max(parMs, 1)))
}
