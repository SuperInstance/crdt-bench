const std = @import("std");

const N_NODES: usize = 32;
const BLOOM_WORDS: usize = 94;
const SKETCH_D: usize = 7;
const SKETCH_W: usize = 1000;
const N_RUNS: usize = 2_000_000;

fn gcounter_merge_simd(dst: *[N_NODES]u64, src: *const [N_NODES]u64) void {
    const Vec16 = @Vector(16, u64);
    const half = N_NODES / 2;
    const dst_lo: Vec16 = dst[0..half].*;
    const src_lo: Vec16 = src[0..half].*;
    dst[0..half].* = @max(dst_lo, src_lo);
    const dst_hi: Vec16 = dst[half..].*;
    const src_hi: Vec16 = src[half..].*;
    dst[half..].* = @max(dst_hi, src_hi);
}

fn gcounter_merge_scalar(dst: *[N_NODES]u64, src: *const [N_NODES]u64) void {
    for (dst, src) |*d, s| d.* = @max(d.*, s);
}

fn bloom_merge_simd(dst: *[BLOOM_WORDS]u64, src: *const [BLOOM_WORDS]u64) void {
    const Vec8 = @Vector(8, u64);
    comptime var ci: usize = 0;
    inline while (ci + 8 <= BLOOM_WORDS) : (ci += 8) {
        const d: Vec8 = dst[ci..][0..8].*;
        const s: Vec8 = src[ci..][0..8].*;
        dst[ci..][0..8].* = d | s;
    }
    var i: usize = ci;
    while (i < BLOOM_WORDS) : (i += 1) dst[i] |= src[i];
}

fn bloom_merge_scalar(dst: *[BLOOM_WORDS]u64, src: *const [BLOOM_WORDS]u64) void {
    for (dst, src) |*d, s| d.* |= s;
}

fn sketch_merge_simd(dst: *[SKETCH_D][SKETCH_W]u64, src: *const [SKETCH_D][SKETCH_W]u64) void {
    const Vec8 = @Vector(8, u64);
    for (0..SKETCH_D) |d| {
        comptime var w: usize = 0;
        inline while (w + 8 <= SKETCH_W) : (w += 8) {
            const dv: Vec8 = dst[d][w..][0..8].*;
            const sv: Vec8 = src[d][w..][0..8].*;
            dst[d][w..][0..8].* = @max(dv, sv);
        }
        while (w < SKETCH_W) : (w += 1) dst[d][w] = @max(dst[d][w], src[d][w]);
    }
}

fn sketch_merge_scalar(dst: *[SKETCH_D][SKETCH_W]u64, src: *const [SKETCH_D][SKETCH_W]u64) void {
    for (0..SKETCH_D) |d| {
        for (0..SKETCH_W) |w| dst[d][w] = @max(dst[d][w], src[d][w]);
    }
}

var gc_a: [N_NODES]u64 = undefined;
var gc_b: [N_NODES]u64 = undefined;
var bl_a: [BLOOM_WORDS]u64 = undefined;
var bl_b: [BLOOM_WORDS]u64 = undefined;
var sk_a: [SKETCH_D][SKETCH_W]u64 = undefined;
var sk_b: [SKETCH_D][SKETCH_W]u64 = undefined;

fn init() void {
    for (0..N_NODES) |i| { gc_a[i] = @intCast(i * i); gc_b[i] = 100; }
    for (0..BLOOM_WORDS) |i| { bl_a[i] = @intCast(i * i * i); bl_b[i] = 0xAAAAAAAAAAAAAAAA; }
    for (0..SKETCH_D) |d| for (0..SKETCH_W) |w| {
        sk_a[d][w] = @intCast((d + 1) * (w + 1));
        sk_b[d][w] = @intCast((d + 2) * (w + 3));
    };
}

pub fn main() !void {
    init();
    const w = std.debug.print;

    w("╔═══════════════════════════════════════════════════════════════╗\n", .{});
    w("║  CRDT MERGE — Zig 0.16 (Zen 5 + AVX-512)                   ║\n", .{});
    w("╚═══════════════════════════════════════════════════════════════╝\n\n", .{});

    // G-Counter
    run("G-Counter: @Vector(16,u64) SIMD max", struct { fn f() void { gcounter_merge_simd(&gc_a, &gc_b); } }.f);
    run("G-Counter: scalar loop @max", struct { fn f() void { gcounter_merge_scalar(&gc_a, &gc_b); } }.f);
    w("\n", .{});

    // Bloom
    run("Bloom: @Vector(8,u64) SIMD OR", struct { fn f() void { bloom_merge_simd(&bl_a, &bl_b); } }.f);
    run("Bloom: scalar loop |=", struct { fn f() void { bloom_merge_scalar(&bl_a, &bl_b); } }.f);
    w("\n", .{});

    // Sketch
    run("Sketch: @Vector(8) SIMD max (7K)", struct { fn f() void { sketch_merge_simd(&sk_a, &sk_b); } }.f);
    run("Sketch: scalar loop max (7K)", struct { fn f() void { sketch_merge_scalar(&sk_a, &sk_b); } }.f);

    w("\n--- Zig @Vector SIMD analysis ---\n", .{});
    w("  @Vector(16, u64): maps to vpmaxuq (AVX-512) on Zen 5\n", .{});
    w("  @Vector(8, u64):  maps to vor (AVX-512) for Bloom OR\n", .{});
    w("  comptime inline: loop unrolling at compile time, zero overhead\n", .{});
}

const linux = std.os.linux; var _ts: linux.timespec = undefined;
fn clock_ns() i128 {
    _ = linux.clock_gettime(.MONOTONIC, &_ts); // CLOCK_MONOTONIC
    return @as(i128, _ts.sec) * 1_000_000_000 + _ts.nsec;
}

fn run(comptime label: []const u8, comptime f: fn () void) void {
    for (0..1000) |_| f();
    const start = clock_ns();
    for (0..N_RUNS) |_| f();
    const elapsed = clock_ns() - start;
    const ns: f64 = @floatFromInt(@divTrunc(elapsed, @as(i128, @intCast(N_RUNS))));
    if (ns > 0) {
        std.debug.print("  {s:<45} {:>8.1} ns  {:>12.0} ops/s\n", .{ label, ns, 1e9 / ns });
    } else {
        std.debug.print("  {s:<45}    <1.0 ns  (optimized away?)\n", .{label});
    }
}
