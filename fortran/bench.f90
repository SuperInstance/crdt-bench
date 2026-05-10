! CRDT Merge — Fortran (fixed: prevent optimizer from eliminating small arrays)
program crdt_bench
  use iso_fortran_env, only: int64, real64
  implicit none

  integer, parameter :: N_NODES = 32
  integer, parameter :: BLOOM_WORDS = 94
  integer, parameter :: SKETCH_D = 7
  integer, parameter :: SKETCH_W = 1000
  integer, parameter :: N_RUNS = 2000000

  integer(int64) :: gc_a(N_NODES), gc_b(N_NODES)
  integer(int64) :: bl_a(BLOOM_WORDS), bl_b(BLOOM_WORDS)
  integer(int64) :: sk_a(SKETCH_D, SKETCH_W), sk_b(SKETCH_D, SKETCH_W)
  integer(int64) :: sink

  real(real64) :: start_time, end_time, ns_per
  integer :: i, d, w

  print *, "=== CRDT MERGE - Fortran (gfortran -O3 -march=native, Zen 5) ==="
  print *, ""

  ! Init
  do i = 1, N_NODES
    gc_a(i) = int(i * i, int64)
    gc_b(i) = 100_int64
  end do
  do i = 1, BLOOM_WORDS
    bl_a(i) = int(i * i * i, int64)
    bl_b(i) = int(z'AAAAAAAAAAAAAAAA', int64)
  end do
  do d = 1, SKETCH_D
    do w = 1, SKETCH_W
      sk_a(d, w) = int(d * w, int64)
      sk_b(d, w) = int((d+1) * (w+2), int64)
    end do
  end do

  ! ---- G-Counter: whole-array MAX ----
  sink = 0
  call cpu_time(start_time)
  do i = 1, N_RUNS
    gc_a = max(gc_a, gc_b)
    sink = sink + gc_a(1)  ! Prevent elimination
  end do
  call cpu_time(end_time)
  ns_per = (end_time - start_time) / N_RUNS * 1e9
  write(*,'(A,F8.1,A,F12.0,A)') "  G-Counter: whole-array MAX       ", ns_per, " ns  ", 1e9/ns_per, " ops/s"

  ! Explicit loop
  sink = 0
  call cpu_time(start_time)
  do i = 1, N_RUNS
    call gc_loop(gc_a, gc_b)
    sink = sink + gc_a(1)
  end do
  call cpu_time(end_time)
  ns_per = (end_time - start_time) / N_RUNS * 1e9
  write(*,'(A,F8.1,A,F12.0,A)') "  G-Counter: explicit DO loop      ", ns_per, " ns  ", 1e9/ns_per, " ops/s"

  print *, ""

  ! ---- Bloom: whole-array IOR ----
  sink = 0
  call cpu_time(start_time)
  do i = 1, N_RUNS
    bl_a = ior(bl_a, bl_b)
    sink = sink + bl_a(1)
  end do
  call cpu_time(end_time)
  ns_per = (end_time - start_time) / N_RUNS * 1e9
  write(*,'(A,F8.1,A,F12.0,A)') "  Bloom: whole-array IOR           ", ns_per, " ns  ", 1e9/ns_per, " ops/s"

  ! Explicit loop
  sink = 0
  call cpu_time(start_time)
  do i = 1, N_RUNS
    call bl_loop(bl_a, bl_b)
    sink = sink + bl_a(1)
  end do
  call cpu_time(end_time)
  ns_per = (end_time - start_time) / N_RUNS * 1e9
  write(*,'(A,F8.1,A,F12.0,A)') "  Bloom: explicit DO loop          ", ns_per, " ns  ", 1e9/ns_per, " ops/s"

  print *, ""

  ! ---- Sketch: whole-array MAX ----
  sink = 0
  call cpu_time(start_time)
  do i = 1, N_RUNS
    sk_a = max(sk_a, sk_b)
    sink = sink + sk_a(1, 1)
  end do
  call cpu_time(end_time)
  ns_per = (end_time - start_time) / N_RUNS * 1e9
  write(*,'(A,F8.1,A,F12.0,A)') "  Sketch: 2D whole-array MAX (7K)  ", ns_per, " ns  ", 1e9/ns_per, " ops/s"

  print *, ""
  print *, "  Fortran: CRDT merge in 1 line of code."
  print *, "  dst = MAX(dst, src)   -- G-Counter"
  print *, "  dst = IOR(dst, src)   -- Bloom filter"
  print *, "  dst = MAX(dst, src)   -- Sketch"
  print *, "  Compiler does the rest (AVX-512 auto-vectorization)."

  ! Use sink to prevent complete elimination
  if (sink < 0) print *, sink

contains

  subroutine gc_loop(dst, src)
    integer(int64), intent(inout) :: dst(:)
    integer(int64), intent(in) :: src(:)
    integer :: j
    do j = 1, size(dst)
      if (src(j) > dst(j)) dst(j) = src(j)
    end do
  end subroutine

  subroutine bl_loop(dst, src)
    integer(int64), intent(inout) :: dst(:)
    integer(int64), intent(in) :: src(:)
    integer :: j
    do j = 1, size(dst)
      dst(j) = ior(dst(j), src(j))
    end do
  end subroutine

end program crdt_bench
