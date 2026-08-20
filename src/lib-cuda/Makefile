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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 32-bit index compile policy  (see core/dispatch.h)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The same idea as BOUNDFLAGS/SPLINEFLAGS, one axis further out again, and the
# most expensive of the three. EVERY templated kernel below this layer is
# templated on `offset_t`, whose two values (int32_t / int64_t) are chosen per
# call by canUse32BitIndexMath, so the narrow path costs exactly x2
# instantiations of everything -- which here means x2 device code, x2 SASS and
# x2 ptxas memory:
#
#   make INDEXFLAGS="-DFF_INDEX32=0"     # one offset type (int64_t) everywhere
#
# ~~~ MEASURED ON THIS BACKEND, 2026-08-20 (fastfields-lib#143) ~~~
# Both positions built by the `build-cuda` matrix in one CI run, nvcc -O1 -j2
# with the BOUNDFLAGS/SPLINEFLAGS defaults above, on the same ubuntu-latest
# runner class. Peak nvcc RSS (GiB) and wall time per module:
#
#                        FF_INDEX32=1        FF_INDEX32=0        RSS delta
#     reg_flow            12.97  1023 s       5.69   497 s        -56.1%
#     reg_field            8.09   791 s       3.76   343 s        -53.5%
#     reg_field_rls        7.07   701 s       3.42   353 s        -51.7%
#     resize               2.00   193 s       1.08   113 s        -46.2%
#     reg_flow_rls         1.90   155 s       1.01    89 s        -46.9%
#     pushpull_backward    1.51   167 s       0.80   102 s        -46.9%
#     pushpull             1.48   158 s       0.79    97 s        -46.6%
#     restrict             1.30   120 s       0.68    72 s        -47.4%
#     distance             0.80   118 s       0.54    61 s        -33.1%
#     splinc               0.42    68 s       0.27    37 s        -35.5%
#     posdef               0.37    40 s       0.35    28 s        -6.9%
#     ---------------------------------------------------------------------
#     total nvcc CPU      3532 s              1791 s              -49.3%
#     job wall clock       ~38 min             ~19 min
#
# So on CUDA the axis costs a little over half of the heaviest module and just
# under half the whole compile. (For reference, the host compiler on the same
# sources: reg_flow 10,219 -> 5,123 template instantiations, -49.9%.)
#
# Two caveats on the left-hand column. `reg_flow` reproduces the 12.98 figure
# recorded below to 0.1%, but `reg_field` came in at 8.09 rather than the
# recorded 8.93 -- the "better than 0.1%" note under that table holds for
# reg_flow and does not appear to hold for reg_field. And the budget check is
# per module, so the -j2 overlap risk described below is unchanged by any of
# this: halving each peak halves the sums too, but only if the axis is actually
# turned off, which by default it is not.
#
# It is nevertheless ON by default, which is a deliberate choice and not
# inertia. Narrow indices are a register-pressure optimisation inherited from
# ATen and there is more reason to expect them to pay on a GPU than on a CPU
# -- but nobody has ever measured it here, because there is no GPU in CI. The
# honest reading of the numbers above is therefore uncomfortable: this backend
# is the one with no headroom left, and it is also the one keeping the axis. If
# a benchmark on real hardware ever shows the narrow path buys nothing (or less
# than the per-call cudaMallocHost in core/autocast.h costs it), flipping this
# default takes reg_flow from 12.97 GiB to 5.69 and the whole compile from ~38
# to ~19 minutes. Until then the knob exists, both positions are built by CI,
# and the default does not move. What would settle it: fastfields-lib#144.
#
# Kept out of CXXFLAGS for the same reason as BOUNDFLAGS/SPLINEFLAGS.
INDEXFLAGS  ?= -DFF_INDEX32=1

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
# reg_flow was the 12.98 GiB entry in the table above -- 81% of a 16 GB runner,
# in one indivisible TU, and the single number that pinned this build at -j2.
# It is now twelve instantiation slices plus a thin front TU (reg_flow.cpp),
# which keeps every exported symbol and instantiates nothing. See
# reg_flow_slice.h for the seam and reg_flow_slice.inl for the bodies.
#
# The cut is (operation family) x (ndim), which is how the dispatch already
# factors: each entry point selects one internal wrapper template and
# `switch (ndim)`es it, and no two arms share an instantiation. The exported
# ABI is unchanged -- the slice symbols are hidden-visibility and never reach
# .dynsym.
#
# ~~~ MEASURED, build-cuda on #147 vs on 85fdac7 (same image, nvcc 12.0.140,
#     same -O1, same bound/spline policy, index32 leg) ~~~
#
#     reg_flow_matvec_3d   1.95 GiB   174.90 s   reg_flow_matvec_1d  0.61  43.46
#     reg_flow_matvec_2d   1.22 GiB    91.61 s   reg_flow_relax_2d   0.60  45.85
#     reg_flow_diag_3d     1.03 GiB    59.12 s   reg_flow_kernel_2d  0.59  44.29
#     reg_flow_relax_3d    0.96 GiB    87.40 s   reg_flow_kernel_1d  0.48  33.84
#     reg_flow_kernel_3d   0.89 GiB    78.43 s   reg_flow_diag_1d    0.44  30.52
#     reg_flow_diag_2d     0.67 GiB    41.22 s   reg_flow_relax_1d   0.26  19.80
#                                                reg_flow (front)    0.20   1.51
#
#   heaviest TU  12.98 -> 1.95 GiB   (-85.0%)
#   module CPU 1149.89 -> 751.95 s   (-34.6%)
#   wall -j2      2243 -> 1672 s     (-25.5%, whole `make cuda`)
#   .so link      1.38 -> 1.05 s
#
# THE COUNTER-FORCE IS REAL BUT SMALL, and was measured rather than assumed.
# Every slice re-parses the whole template header set, so total CPU could have
# risen. A TU that parses impl/cuda/reg_flow.h plus the kernels and
# instantiates nothing costs 0.22 GiB and 1.35 s; twelve extra TUs is therefore
# ~16 s against a 1149.89 s baseline -- 1.4%, an order of magnitude under the
# instantiation work that now divides. That ratio is what makes this split pay.
# It is not a general law: a module whose per-TU instantiation work is
# comparable to 1.35 s should not be split.
#
# WHAT THIS DOES *NOT* BUY: A HIGHER -j.
# The ceiling is max(peak) over EVERY module, and reg_field (8.23 GiB) and
# reg_field_rls (7.07 GiB) are untouched. reg_flow simply stops being the
# constraint; reg_field becomes it, and the reg_field/reg_field_rls overlap
# (15.31 GiB) is the same -j2 hazard described above. Modelling make's dispatch
# over the measured table -- the model reproduces the observed -j2 wall to
# 0.25% -- gives -j4 a worst coincidence of 19.26 GiB, still over a 16 GB box.
# So do NOT raise -j on the strength of this change. Splitting reg_field and
# reg_field_rls the same way (they are structurally identical files) is what
# would take max(peak) to ~2 GiB and make -j4 CPU-bound instead of RAM-bound.
#
# INTERACTION WITH INDEXFLAGS: they MULTIPLY, they do not overlap. Measured on
# the heaviest slice, same TU, index axis toggled:
#
#     reg_flow_matvec_3d  FF_INDEX32=1   1.94 GiB   277.1 s   17.0 MB
#                         FF_INDEX32=0   1.05 GiB   174.4 s   10.0 MB
#                                        -45.8%     -37.1%    -41.1%
#
# That -45.8% is the same proportion #94 measured on the UNSPLIT file with the
# host compiler (-44.6%), which is the point: splitting does not make the index
# axis any cheaper in relative terms, it only shrinks the absolute number the
# proportion applies to. The two compose -- 12.98 -> 1.94 GiB from the split,
# 1.94 -> 1.05 GiB from the knob. So this split is NOT a reason to stop caring
# about FF_INDEX32; what it changes is which resource the knob buys back, since
# memory is no longer what binds this module and wall-clock is.
#
# AND THE TWO TOGETHER ARE WHAT WOULD ACTUALLY UNLOCK -j4. build-cuda's
# index64 leg builds this same split with the axis off, and that combination --
# not either half alone -- is the first configuration in which no module is
# heavy any more:
#
#     reg_field           3.76 GiB   264.90 s     <-- the new maximum
#     reg_field_rls       3.42 GiB   279.38 s
#     resize              1.08 GiB    90.20 s
#     reg_flow_matvec_3d  1.05 GiB    81.45 s     <-- heaviest reg_flow slice
#     ... every other module below 1.01 GiB ...
#     reg_flow (front)    0.20 GiB     0.88 s
#
#   total nvcc CPU 1380.3 s; `make cuda -j2` wall 723 s (measured).
#
# Feeding that table through the same scheduling model (which reproduces the
# measured -j2 wall to 0.5%): -j4 gives a 393 s makespan with a worst
# coincidence of 9.31 GiB, comfortably inside a 16 GB runner. So `-j4` is safe
# on the index64 leg TODAY, with reg_field untouched. It is NOT safe on the
# index32 leg, which is the default and the one that ships -- there the same
# model says 19.26 GiB. Do not raise -j globally on the strength of the
# index64 number.
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
	$(NVCC) $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) $(INDEXFLAGS) $(INCLUDES) \
	  -x cu -Xcompiler -fPIC -MMD -MF $(@:.$(MOSUF)=.d) -c -o $@ $<

# Compile-only probe for the CUDA impl layer: there is no GPU in CI, so the
# mesh launcher is validated by nvcc accepting it, not by running it.
PROBESRC = $(wildcard $(ROOTDIR)/tests/impl-cuda/*.cu)
test-probe: | $(TESTDIR)
	@ for p in $(PROBESRC); do \
	    echo "compiling $$p"; \
	    $(NVCC) $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) $(INDEXFLAGS) $(INCLUDES) \
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
