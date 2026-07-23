# fastfields-lib

**Fast routines for dense scalar and vector fields — a C++/CUDA library.**

`fastfields-lib` is the **hub** of the fastfields project. It compiles to
**`libfastfields.so`** and exposes the public C++ API in the `ff::` namespace.
Every public function takes [DLPack](https://dmlc.github.io/dlpack/latest/)
tensors (`DLTensor`), inspects each tensor's device, and dispatches to the CPU
or CUDA backend library — so callers work against a single device-agnostic
entry point.

fastfields is a rewrite of the JIT-compiled `jitfields`, dropping the
cppyy/torchlib dependency. Data crosses backends via DLPack, and higher layers
bind the library with [nanobind](https://github.com/wjakob/nanobind).

## The layered hierarchy

fastfields is split into small, single-responsibility layers. `fastfields-lib`
sits at the device-dispatch boundary:

```
kernels ─ cpu-impl ─ cpu-lib ┐
        ─ cuda-impl ─ cuda-lib ┴─ lib ← (you are here) ─ bind-py ─ {numpy,cupy,torch} ─ fastfields
```

- **kernels** — voxelwise, header-only, templated single-element math.
- **cpu-impl / cuda-impl** — loops over elements (CPU thread pool / CUDA
  kernels); templated, dynamic sizes.
- **cpu-lib / cuda-lib** — exported symbols taking unsafe pointers; dispatch on
  dtype (and dim/spline/bound) to the templated impl.
- **lib** (this repo) — exported symbols taking `DLTensor`; dispatch on
  `device.device_type` to the cpu or cuda lib.
- **bind-py** and the `{numpy, cupy, torch}` frontends — Python bindings.

See [Architecture](architecture.md) for the design rationale, and
[API families](api.md) for the operations exposed at this level.

## Install / build

`fastfields-lib` builds with a clang-style Makefile (C++11):

```bash
make -C . all CXX=clang++      # builds libfastfields.so; also builds + installs libfastfields-cpu.so
```

`libfastfields.so` links `-lfastfields-cpu` with an `$ORIGIN/../lib` rpath. The
CUDA backend is compile/link-only in CI (no GPU) and is guarded at build time by
`FF_WITH_CUDA`; the CPU path is the source of truth.

The submodule symlinks must exist so the include nesting resolves
(`lib/cpu -> cpu-lib`, `cpu-lib/impl -> cpu-impl`, `cpu-impl/kernels -> kernels`).

There are no standalone tests at this level — correctness is gated by
`fastfields-cpu-lib`'s CPU test suite.
