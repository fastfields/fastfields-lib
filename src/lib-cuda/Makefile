# libfastfields-cuda.so -- the CUDA dtype-dispatch layer.
#
# Every source here is a .cpp compiled by nvcc with `-x cu`; there are no .cu
# files. Nothing in `make cpu` or `make lib` reaches this Makefile unless
# USE_CUDA=1 asks for it.

GROUP := lib-cuda
include ../../make/common.mk

# nvcc flags: -std/-O3 pass straight through. The host compiler's diagnostic
# flags (-ferror-limit / -fmax-errors / -ftemplate-backtrace-limit) are NOT
# understood by nvcc, so DIAGFLAGS is deliberately not added here. nvcc 12
# builds these CUDA headers under -std=c++14.
CXXFLAGS += -std=c++14 -O3
TESTFLAGS += $(DIAGFLAGS)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Boundary-condition compile policy  (see impl/kernels/bounds.h)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Every boundary condition used to be a compile-time template parameter, so
# each regulariser kernel was instantiated 8 times per (ndim, dtype, offset_t)
# combination. On the CUDA side that pushes `ptxas` past 16 GB of RAM -- it gets
# OOM-killed on a standard 16 GB CI runner.
#
# `bound::type::Dynamic` provides the same operator with the condition read at
# run time: one instantiation shared by every condition that does not keep a
# static fast path. The default below keeps dedicated instantiations for two
# conditions and routes the other six through the Dynamic implementation:
#   DCT2 -- Neumann, the library-wide default boundary condition, and the
#           symmetric case (no sign flip at the boundary);
#   DST2 -- Dirichlet, the antisymmetric case, so the sign-flipping code path
#           (`_sign::periodic2` + the sign-aware `cget`/`add` selected by
#           FF_ISO_SIGN) also keeps a real compile-time instantiation rather
#           than only ever being reached through the shared Dynamic one.
#
# Whoever compiles the library can trade compile cost against per-voxel speed:
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=1"                     # all 8 static
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0"                     # all dynamic
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DFT=1 \
#                    -DFF_STATIC_BOUND_DCT2=1"                 # pick your own
# Results are identical either way; only code size, compile cost and speed move.
# Kept out of CXXFLAGS on purpose so that overriding CXXFLAGS on the command
# line (as CI does, to force -O1) does not silently drop the policy.
BOUNDFLAGS  ?= -DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DCT2=1 -DFF_STATIC_BOUND_DST2=1

# Interpolation-order compile policy (see impl/kernels/spline.h) -- the same
# idea one axis further out. `pushpull` templates on the spline order as well as
# the boundary condition (and ndim, dtype, offset_t), so all eight orders static
# would multiply the already-Dynamic-bound matrix by up to eight again.
#
# `spline::type::Dynamic` provides the same operator with the order read at run
# time. The default below keeps dedicated instantiations for the four lowest
# (and by far the most commonly used) orders -- Nearest/Linear/Quadratic/Cubic
# -- and routes FourthOrder-SeventhOrder through the Dynamic implementation:
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=1"                    # all 8 static
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=0"                    # all dynamic
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=0 \
#                     -DFF_STATIC_SPLINE_CUBIC=1"               # pick your own
# Results are identical either way; only code size, compile cost and speed move.
# Kept out of CXXFLAGS for the same reason as BOUNDFLAGS.
SPLINEFLAGS ?= -DFF_STATIC_SPLINES=0 \
               -DFF_STATIC_SPLINE_NEAREST=1 -DFF_STATIC_SPLINE_LINEAR=1 \
               -DFF_STATIC_SPLINE_QUADRATIC=1 -DFF_STATIC_SPLINE_CUBIC=1

# The regularisers are split into a core module and an `_rls` module (the
# reweighted-least-squares ops) -- a CUDA-only split that lib-cpu does not need.
# It is a memory/parallelism measure, not a correctness one: the
# boundary-condition policy above is what brought this build back from a ~16 GB
# ptxas OOM, and CI additionally passes -O1 for the same reason.
#
# ~~~ MEASURED, 2026-08-19 (fastfields-lib#80) ~~~
# The figures this comment used to quote -- "~3.8 GB per split module, ~6-7 GB
# combined" -- are WRONG, and wrong in the dangerous direction. build-cuda now
# wraps nvcc in /usr/bin/time and prints peak RSS per module; on a 16 GB
# ubuntu-latest runner, nvcc -O1 with the BOUNDFLAGS/SPLINEFLAGS defaults above:
#
#     reg_flow            12.98 GB   1097 s      <-- not ~3.8
#     reg_field            8.93 GB    840 s      <-- not ~3.8
#     reg_field_rls        6.67 GB    759 s      <-- not ~3.8
#     resize               2.00 GB    188 s
#     reg_flow_rls         1.90 GB    153 s
#     pushpull_backward    1.51 GB    163 s
#     pushpull             1.48 GB    153 s
#     restrict             1.30 GB    113 s
#     distance             0.77 GB     72 s
#     splinc               0.42 GB     67 s
#     posdef               0.37 GB     34 s
#
# (Peak RSS of the largest single process in the nvcc tree -- cicc or ptxas --
# not a sum: `time` reports ru_maxrss, which is a maximum. Across two runs the
# heavy modules reproduce to better than 0.1%; the small ones wander a few
# percent, which is why the budget below is set per module and with room.)
#
# (Re-derive with: the `build-cuda` job in .github/workflows/ci.yml, step
# "Peak nvcc memory per module". It wraps nvcc in /usr/bin/time on every run.)
#
# ~~~ AND -j2 IS LUCK, NOT HEADROOM ~~~
# Worth being blunt about, because the old numbers made it look settled. Feed
# the measured wall times above through make's -j2 dispatch order (which is
# MODULES order) and the heavy modules do NOT take turns -- they overlap:
#
#     reg_field + reg_field_rls   overlap ~760 s   peaks sum to 15.59 GB
#     reg_field + reg_flow        overlap  ~50 s   peaks sum to 21.90 GB
#     reg_flow  + reg_flow_rls    overlap ~150 s   peaks sum to 14.88 GB
#
# On a 16 GB runner the second of those is over the limit outright and the
# first clears it by ~0.4 GB. The build is green anyway because a module's peak
# RSS is a brief spike late in its own ptxas phase, and so far two spikes have
# not landed in the same instant. Nothing enforces that. Anything that shifts
# the relative phase of two heavy compiles -- reordering MODULES, a different
# nvcc, ccache, a bound/spline policy change, adding -arch/-gencode targets --
# reshuffles it, and the failure mode is an OOM kill that looks like a random
# infrastructure flake rather than a build decision.
#
# So: do NOT raise -j, do NOT drop -O1, and do NOT read a green build as proof
# that two regularisers fit side by side -- they do not. The `build-cuda` job
# fails if any single module exceeds FF_MEM_BUDGET_KB (14 GB), which turns the
# approach to the ceiling into a diagnosable error instead of a mystery OOM;
# it does not and cannot catch two peaks coinciding.
#
# What is NOT in question:
#   * The SPLIT IS STILL LOAD-BEARING and must not be undone -- reg_field and
#     reg_flow are the two heaviest things here even after being halved.
#   * Nothing here is caused by the four modules added below; the regularisers
#     were always the hogs and are untouched by that change.
# Fixing the headroom is out of scope for #80 -- it is a pre-existing
# condition, now measured instead of guessed.
#
# MODULES must list every .cpp in this directory. It did not until
# fastfields-lib#80: posdef, resize, restrict and splinc were present as
# sources and shipped as headers, but never compiled and never linked, so the
# eleven FF_CUDA:: entry points they define (the whole Posdef family,
# `resample`, `restriction`, `spline_coeff`) were undefined symbols in
# libfastfields-cuda.so while every build reported success. The hub's CUDA link
# now passes $(NO_UNDEFINED) (see src/lib/Makefile), so a repeat of that
# omission fails the link instead of surfacing at run time on a GPU.
#
# Adding them does not move the ceiling: in the table above the four are the
# four cheapest modules in the file (resize 2.00, restrict 1.30, splinc 0.42,
# posdef 0.37 GB), together well under a single regulariser. They cost ~7 min
# of the ~31 min compile.
# ~~~ reg_flow IS SPLIT ACROSS TRANSLATION UNITS (fastfields-lib#147) ~~~
#
# `reg_flow` was the 12.98 GiB entry in the table above -- 81% of a 16 GB
# runner, in one indivisible TU, and the single number that pinned this build
# at -j2. It is now twelve instantiation slices plus a thin front TU
# (`reg_flow.cpp`), which keeps every exported symbol and instantiates nothing;
# see reg_flow_slice.h for the seam and reg_flow_slice.inl for the bodies.
#
# The cut is (operation family) x (ndim), which is exactly how the dispatch
# already factors: each entry point selects one internal wrapper template and
# `switch (ndim)`es it, and no two arms share an instantiation. So a slice pays
# for its own arm and nothing else. The exported ABI is unchanged -- the slice
# symbols are hidden-visibility and never reach .dynsym.
#
# ~~~ MEASURED, 2026-08-20, build-cuda on this branch vs on 85fdac7 ~~~
# Same job, same runner image, same nvcc 12.0.140, same -O1 and the
# BOUNDFLAGS/SPLINEFLAGS defaults above. Before, reg_flow was one TU:
#
#     reg_flow            12.98 GiB   1149.89 s        <-- one indivisible TU
#
# After, it is thirteen (peak RSS / nvcc seconds / role):
#
#     reg_flow_matvec_3d   1.95 GiB    174.90 s
#     reg_flow_matvec_2d   1.22 GiB     91.61 s
#     reg_flow_diag_3d     1.03 GiB     59.12 s
#     reg_flow_relax_3d    0.96 GiB     87.40 s
#     reg_flow_kernel_3d   0.89 GiB     78.43 s
#     reg_flow_diag_2d     0.67 GiB     41.22 s
#     reg_flow_matvec_1d   0.61 GiB     43.46 s
#     reg_flow_relax_2d    0.60 GiB     45.85 s
#     reg_flow_kernel_2d   0.59 GiB     44.29 s
#     reg_flow_kernel_1d   0.48 GiB     33.84 s
#     reg_flow_diag_1d     0.44 GiB     30.52 s
#     reg_flow_relax_1d    0.26 GiB     19.80 s
#     reg_flow             0.20 GiB      1.51 s        <-- the front TU
#                          -------      --------
#     heaviest slice       1.95 GiB     751.95 s total
#
#   peak      12.98 -> 1.95 GiB   (-85.0%; the heaviest slice)
#   CPU     1149.89 -> 751.95 s   (-34.6%, for this module)
#   wall -j2  2243 -> 1672 s      (-25.5%, whole `make cuda`)
#   link      1.38 -> 1.05 s      (the .so link does not care)
#
# THE COUNTER-FORCE IS REAL BUT SMALL, and it was measured rather than
# assumed. Every slice re-parses the whole template header set, so total CPU
# could have gone UP. A TU that parses `impl/cuda/reg_flow.h` plus the kernels
# and instantiates nothing costs 0.22 GiB and 1.35 s. Twelve extra TUs is
# therefore ~16 s of re-parsing against a 1149.89 s baseline -- 1.4%, an order
# of magnitude less than the instantiation work that now divides. That ratio
# is what makes this split pay; it is not a general law, and a module whose
# per-TU instantiation work is comparable to 1.35 s should not be split.
#
# WHAT THIS DOES *NOT* BUY: A HIGHER -j.
# The ceiling is max(peak) over EVERY module, and reg_field (8.23 GiB) and
# reg_field_rls (7.07 GiB) are untouched. reg_flow is simply no longer the
# constraint -- reg_field is, and their overlap (15.31 GiB) is the same -j2
# hazard the section above describes. Modelling make's dispatch over the
# measured table (the model reproduces the observed -j2 wall to 0.25%):
#
#     -j2   1668 s   worst coincidence 15.31 GiB   reg_field+reg_field_rls
#     -j4    869 s   worst coincidence 19.26 GiB   <-- still over a 16 GB box
#
# So do NOT raise -j on the strength of this change. Splitting reg_field and
# reg_field_rls the same way -- they are structurally identical files, see
# reg_flow_slice.h -- is what would take max(peak) to ~2 GiB and make -j4
# CPU-bound instead of RAM-bound.
#
# ORDER MATTERS HERE. make dispatches in MODULES order, so the slices are
# listed heaviest-first (3D, then 2D, then 1D): with the long poles started
# first, the short ones backfill the tail instead of extending it.
REG_FLOW_SLICES = \
	reg_flow_matvec_3d \
	reg_flow_kernel_3d \
	reg_flow_relax_3d \
	reg_flow_diag_3d \
	reg_flow_matvec_2d \
	reg_flow_kernel_2d \
	reg_flow_relax_2d \
	reg_flow_diag_2d \
	reg_flow_matvec_1d \
	reg_flow_kernel_1d \
	reg_flow_relax_1d \
	reg_flow_diag_1d

MODULES = \
	distance \
	posdef \
	resize \
	restrict \
	splinc \
	reg_field \
	reg_field_rls \
	reg_flow \
	$(REG_FLOW_SLICES) \
	reg_flow_rls \
	pushpull \
	pushpull_backward

OBJECTS = $(addprefix $(OBJDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
TARGET  = $(LIBDIR)/libfastfields-cuda.$(SOSUF)

.PHONY: all install test-probe clean clean-obj clean-lib

all: verb.build.lib $(TARGET) verb.build.lib.done

install: all

$(TARGET): $(OBJECTS) | $(LIBDIR)
	$(NVCC) $(CXXFLAGS) -shared -Xcompiler -fPIC \
	  -Xlinker -$(SONAME)=libfastfields-cuda.$(SOSUF) -o $@ $^

# Header dependency files, so that editing a kernel/impl header rebuilds the
# affected object. This layer had no dependency tracking at all, which in a
# header-only codebase meant stale objects survived header edits.
#
# These MUST be nvcc's own -MMD/-MF, NOT `-Xcompiler -MMD`. nvcc does not hand
# the host compiler the original .cpp: it hands it a generated
# /tmp/tmpxft_*.cudafe1.cpp, so a host-generated .d names that temporary as the
# prerequisite. The temp file is deleted when nvcc exits, so the NEXT `make`
# reads the stale .d and dies with "No rule to make target
# '/tmp/tmpxft_....cudafe1.cpp'". nvcc's own -MMD resolves dependencies against
# the real source and its headers.
#
# nvcc has no -MP equivalent, so a *deleted* header still breaks the build until
# the stale .d is removed -- the tradeoff for having any dependency tracking
# here at all. lib-cpu, compiled by the host compiler directly, keeps -MP.
$(OBJDIR)/%.$(MOSUF): %.cpp | $(OBJDIR)
	$(NVCC) $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) $(INCLUDES) \
	  -x cu -Xcompiler -fPIC -MMD -MF $(@:.$(MOSUF)=.d) -c -o $@ $<

# Compile-only probe for the CUDA impl layer: there is no GPU in CI, so the
# mesh launcher is validated by nvcc accepting it, not by running it.
PROBESRC = $(wildcard $(ROOTDIR)/tests/impl-cuda/*.cu)
test-probe: | $(TESTDIR)
	@ for p in $(PROBESRC); do \
	    echo "compiling $$p"; \
	    $(NVCC) $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) $(INCLUDES) \
	      -c -o $(TESTDIR)/$$(basename $$p .cu).$(MOSUF) $$p || exit 1; \
	done

-include $(wildcard $(OBJDIR)/*.d)

clean: clean-obj clean-lib
clean-obj:
	$(DEL) -r $(OBJDIR)
clean-lib:
	$(DEL) $(TARGET)

verb.build.lib:
	$(call verb, "Building CUDA library...")
verb.build.lib.done:
	$(call verb, "Building CUDA library: done.")
