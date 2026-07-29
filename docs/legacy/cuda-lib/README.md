# fastfields-cuda-lib

## Boundary-condition compile policy (`BOUNDFLAGS`)

Boundary conditions are template parameters, so the dispatch layer would
instantiate every kernel once per `(ndim x bound x dtype x offset_t x nbatch x
nc)` combination. With all eight conditions static, `ptxas` needs ~16 GB to
compile `reg_field.cpp` / `reg_flow.cpp` and gets OOM-killed on a standard CI
runner.

`bound::type::Dynamic` provides the same operators with the condition read at
run time: one instantiation shared by every condition that does not keep a
static fast path. `BOUNDFLAGS` chooses which do:

```
make                                                    # default: DCT2 static, rest dynamic
make BOUNDFLAGS="-DFF_STATIC_BOUNDS=1"                  # all eight static (needs >16 GB)
make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0"                  # all dynamic
make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DCT2=1 -DFF_STATIC_BOUND_DFT=1"
```

Per-condition macros are `FF_STATIC_BOUND_{ZERO,REPLICATE,DCT1,DCT2,DST1,DST2,
DFT,NOCHECK}`. Results are identical either way — only code size, compile cost
and per-voxel speed change. `BOUNDFLAGS` is deliberately *not* part of
`CXXFLAGS`, so overriding `CXXFLAGS` on the command line cannot silently drop
the policy. See fastfields-lib#43.
