# fastfields -- top-level delegation.
#
#   make cpu     build build/lib/libfastfields-cpu.so   (never needs nvcc)
#   make cuda    build build/lib/libfastfields-cuda.so  (needs nvcc)
#   make lib     build build/libfastfields.so           (implies cpu)
#   make all     the default: cpu + lib
#   make test    run every test group that does not need a GPU toolchain
#   make clean   remove build/

GROUP := root
include make/common.mk

.PHONY: all cpu cuda lib test test-lib test-lib-cpu test-impl-cuda test-kernels \
        test-atomics clean install

# `all` produces both shared objects -- build/lib/libfastfields-cpu.so and
# build/libfastfields.so -- exactly as the hub repo's `all` did. CUDA stays
# opt-in (USE_CUDA=1, or `make cuda`) because CI has no GPU and the CPU path is
# the tested source of truth.
all: lib

cpu:
	$(MAKE) -C src/lib-cpu all

cuda:
	$(MAKE) -C src/lib-cuda all

lib: cpu
	$(MAKE) -C src/lib all

install:
	$(MAKE) -C src/lib install

########################################################################
#	Tests
########################################################################

test: test-lib-cpu test-lib

test-lib-cpu:
	$(MAKE) -C src/lib-cpu test

test-lib:
	$(MAKE) -C src/lib test

# Needs nvcc; not part of `make test`.
test-impl-cuda:
	$(MAKE) -C src/lib-cuda test-probe

# A hand-run scratch program for the vector/ abstractions; compiled (not run)
# as a check that the headers still stand alone.
test-kernels: | $(TESTDIR)
	$(CXX) $(CXXFLAGS) $(DIAGFLAGS) $(INCLUDES) -std=c++11 \
	  -o $(BUILDDIR)/test/kernels_vector tests/kernels/vector/test.cpp

# What ff::anyAtomicAdd is on the CPU, and that concurrent accumulation into
# disjoint slots stays exact. Deliberately NOT part of `make test`: the 13
# suites and 59,886 checks that gate is defined by must not move. It compiles
# and runs in milliseconds and is built by the tsan CI leg, which is where the
# threaded case is worth anything. Honours CXXFLAGS, so
#   make test-atomics CXXFLAGS="-std=c++11 -O1 -g -DFF_GRAIN_SIZE=1 -fsanitize=thread"
# is the interesting invocation.
test-atomics: | $(TESTDIR)
	$(CXX) $(CXXFLAGS) $(DIAGFLAGS) $(INCLUDES) -std=c++11 \
	  -o $(BUILDDIR)/test/kernels_atomic tests/kernels/atomic/test.cpp
	$(BUILDDIR)/test/kernels_atomic

clean:
	$(DEL) -r $(BUILDDIR)
