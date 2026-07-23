# Architecture

fastfields is organized as a stack of thin, single-responsibility layers. Each
layer only knows about the one directly beneath it, and each adds exactly one
concern — iteration, dtype dispatch, or device dispatch — on top of pure
per-element math.

```
kernels (voxelwise, header-only, templated)
  └─ {cpu,cuda}-impl  (loops over elements; CPU thread pool / CUDA kernels; templated, dynamic sizes)
       └─ {cpu,cuda}-lib  (exported symbols; unsafe pointers -> template dispatch on dtype/dim)
            └─ lib  (exported symbols; DLTensor in; dispatch on device -> cpu/cuda lib)
```

## The layers

### kernels
Single-element math only: one voxel, one point, one small matrix. Header-only,
backend-agnostic (the same source compiles for CPU or CUDA), and templated on
data and pointer types. No device loops live here.

### cpu-impl / cuda-impl
The **loop** layer. It owns the iteration over array elements — a CPU thread
pool for `cpu-impl`, `__global__` device kernels plus host launchers for
`cuda-impl` — and calls a kernel per element. Header-only, templated, with
dynamic (runtime) sizes.

### cpu-lib / cuda-lib
The **dtype-dispatch** boundary. Public symbols take `void*`/pointers plus
shapes and strides — no templates leak into the exported ABI. Each function
picks `scalar_t`/`offset_t` (and dim/spline/bound where relevant) and calls the
templated impl. Pointer and stride arrays are narrowed to 32-bit when
`canUse32BitIndexMath` allows (see `autocast.h`). The CUDA variant additionally
forwards a CUDA `stream` to the impl launcher.

### lib (this repo)
The **device-dispatch** boundary. Each public `ff::<fn>(DLTensor&, …)` inspects
the tensor's `device.device_type` and forwards to `FF_CPU::` (`ff::cpu`) or
`FF_CUDA::` (`ff::cuda`) — the latter guarded by `FF_WITH_CUDA`. This is the one
device-agnostic entry point that higher layers and bindings call.

## Why DLPack

Data crosses backend boundaries as [DLPack](https://dmlc.github.io/dlpack/latest/)
tensors (`DLTensor`). DLPack is a small, framework-neutral tensor ABI: it
carries the data pointer, device, dtype, shape, and strides without pulling in
any framework runtime. That lets fastfields:

- accept tensors from NumPy, CuPy, PyTorch, and others without a hard dependency
  on any of them;
- decide CPU-vs-CUDA execution purely from the tensor's `device` field;
- keep the exported ABI free of C++ templates, so the `.so` boundary stays
  stable while the templated impl underneath can specialize freely.

The Python frontends bind this library with nanobind.

## Device dispatch and the CUDA path

At the `lib` layer, dispatch is a switch on the device type. The CPU path is the
source of truth and the primary automated test gate — it links into
`libfastfields-cpu.so` and is validated by `fastfields-cpu-lib`'s test suite
(each module checked against a brute-force reference).

The CUDA path mirrors the CPU path file-for-file, but is currently
**compile/link-only** (no GPU in CI). Some `cuda-impl` host launchers are still
missing, so a subset of modules is intentionally omitted from the cuda-lib
build until those launchers land. Runtime CUDA correctness needs real hardware.

## Public API naming

Three operations are renamed relative to the impl because a namespace cannot
share a name with a function inside `ff::cpu`:

| impl module | public op       |
|-------------|-----------------|
| `resize`    | `resample`      |
| `restrict`  | `restriction`   |
| `splinc`    | `spline_coeff`  |

`restriction` **accumulates into `out`**, so callers must pre-zero the output.
