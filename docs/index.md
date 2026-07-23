# fastfields-lib

**A fast C++/CUDA library for dense scalar and vector fields.**

`fastfields-lib` compiles to `libfastfields.so` and exposes the project's
operations through a single, device-agnostic C++ API (namespace `ff::`). You
hand it a tensor, it runs on whatever device the tensor lives on — CPU or CUDA —
and returns the result. It is the C++ core behind the Python packages; most
users want [`fastfields`](https://fastfields.github.io/) (`pip install`), not
this library directly.

## What it offers

- **Distance transforms** — Euclidean and L1 distance maps; distance from points
  to a 1-D spline or to a triangle mesh.
- **Resampling** — spline interpolation between grids, its adjoint, and spline
  coefficient prefiltering.
- **Pushpull sampling** — gather/scatter at arbitrary coordinates, with spatial
  gradients — the core of image warping.
- **Positive-definite linear algebra** — matrix-vector products, solves and
  inverses over fields of small symmetric matrices.
- **Regularisers** — absolute / membrane / bending energies on multi-channel
  fields and vector flows.

See [API families](api.md) for the operations in more detail and
[Architecture](architecture.md) if you are integrating or extending the library.

## Build

```bash
make -C . all CXX=clang++      # builds libfastfields.so (+ libfastfields-cpu.so)
```

The library is C++11 and builds with a clang-style Makefile. The CUDA backend is
optional (guarded by `FF_WITH_CUDA`); the CPU path is always built.
