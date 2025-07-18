# fastfields-lib-core

This repository contains utilities and core elements of the fast field
algorithms. Specifically, it implements the algorithms' kernels, _i.e._,
functions that compute on single elements. Complete algorithms would
typically apply these single-element functions on all elements in an
array, in parallel fashion.

It is intended to be used as a submodule by repositories that implement
the full algorithms.
