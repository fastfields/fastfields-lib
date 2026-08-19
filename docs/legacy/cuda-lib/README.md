# fastfields-cuda-lib

## Boundary-condition compile policy (`BOUNDFLAGS`)

Boundary conditions are template parameters, so the dispatch layer would
instantiate every kernel once per `(ndim x bound x dtype x offset_t x nbatch x
nc)` combination. With all eight conditions static, `ptxas` needs ~16 GB to
compile `reg_field.cpp` / `reg_flow.cpp` and gets OOM-killed on a standard CI
runner.

`bound::type::Dynamic` provides the same operators with the condition read at
run time: one instantiation shared by every condition that does not keep a
static fast path. The default keeps two static: **DCT2** (Neumann, the
library-wide default, symmetric -- no sign flip) and **DST2** (Dirichlet,
antisymmetric), so the sign-flipping path keeps a real compile-time
instantiation too rather than only ever being exercised through the shared
Dynamic one. `BOUNDFLAGS` chooses which do:

```
make                                                    # default: DCT2+DST2 static, rest dynamic
make BOUNDFLAGS="-DFF_STATIC_BOUNDS=1"                  # all eight static (needs >16 GB)
make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0"                  # all dynamic
make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DCT2=1 -DFF_STATIC_BOUND_DFT=1"
```

Per-condition macros are `FF_STATIC_BOUND_{ZERO,REPLICATE,DCT1,DCT2,DST1,DST2,
DFT,NOCHECK}`. Results are identical either way — only code size, compile cost
and per-voxel speed change. `BOUNDFLAGS` is deliberately *not* part of
`CXXFLAGS`, so overriding `CXXFLAGS` on the command line cannot silently drop
the policy. See fastfields-lib#43.

## Interpolation-order compile policy (`SPLINEFLAGS`)

`pushpull` templates on the spline (interpolation) order in addition to the
boundary condition, `ndim`, `dtype` and `offset_t`. With all eight orders
static, `pushpull.cpp` alone measured **805 s wall / 4.3 GB peak `ptxas`**
(on top of the already-Dynamic-by-default `BOUNDFLAGS` above) — survivable in
isolation, but worth shrinking on top of the regulariser modules under CI's
`-j2` budget.

`spline::type::Dynamic` is the same idea as `bound::type::Dynamic`, one axis
further out: a single runtime-dispatched instantiation shared by every order
that does not keep a static fast path. The default keeps the four lowest (and
by far most commonly used) orders static — **Nearest, Linear, Quadratic,
Cubic** — and routes FourthOrder–SeventhOrder through Dynamic:

```
make                                                     # default: orders 0-3 static, 4-7 dynamic
make SPLINEFLAGS="-DFF_STATIC_SPLINES=1"                 # all eight static (805s/4.3GB for pushpull.cpp alone)
make SPLINEFLAGS="-DFF_STATIC_SPLINES=0"                 # all dynamic
make SPLINEFLAGS="-DFF_STATIC_SPLINES=0 -DFF_STATIC_SPLINE_CUBIC=1"
```

Per-order macros are `FF_STATIC_SPLINE_{NEAREST,LINEAR,QUADRATIC,CUBIC,
FOURTHORDER,FIFTHORDER,SIXTHORDER,SEVENTHORDER}`. Measured with the default
policy: `pushpull.cpp` compiles in **266 s wall / 1.52 GB peak `ptxas`** — a 3x
wall-time and 2.8x memory reduction versus all-static, and comfortably inside
the ~3.8–7 GB per-module envelope the regularisers already established as
CI-survivable under `-j2`. Results are identical either way — only code size,
compile cost and per-voxel speed change. `SPLINEFLAGS` is deliberately *not*
part of `CXXFLAGS`, for the same reason as `BOUNDFLAGS`. See
fastfields-cuda-lib#30.
