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
# with the BOUNDFLAGS/SPLINEFLAGS defaults above, on ubuntu-24.04 with nvcc
# 12.0.140 -- see the image note under MEASURED below: these are that pair's
# figures, and the absolute columns have since moved (the RSS *ratio* between
# the two positions is the load-bearing part here and is what the decision
# rests on). Peak nvcc RSS (GiB) and wall time per module:
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
# Two caveats on the left-hand column. On its own image `reg_flow` reproduced
# the 12.98 figure to 0.1% while `reg_field` came in at 8.09 against 8.93 --
# so even same-image reproducibility was never uniform across modules, and
# `reg_field` was the module it was worst for. (That is also the module the
# 2026-11 image change moved most: 8.35 -> 5.32 GiB. Whatever makes reg_field's
# peak volatile, it is not the source.) And the budget check is per module, so
# the -j2 overlap risk described below is unchanged by any of this: halving
# each peak halves the sums too, but only if the axis is actually turned off,
# which by default it is not.
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
# ~~~ MEASURED -- AND THE MEASUREMENT IS IMAGE-DEPENDENT ~~~
#
# READ THIS BEFORE QUOTING ANY NUMBER BELOW. Every figure here is a
# (runner image, nvcc) pair's figure, not a property of this source tree.
# Each table is therefore stamped with the image and the nvcc it came from,
# and a table whose stamp does not match what CI runs today is history, not a
# specification. The earlier version of this comment said the heavy modules
# "reproduce to better than 0.1% across runs" -- true, and misleading: they
# reproduce to 0.1% across runs *on one image*, and not at all across images.
#
# (Peak RSS of the largest single process in the nvcc tree -- cicc or ptxas --
# not a sum: `time` reports ru_maxrss, which is a maximum.)
#
# (Re-derive with: the `build-cuda` job in .github/workflows/ci.yml, step
# "Peak nvcc memory per module". It wraps nvcc in /usr/bin/time on every run,
# so the current figures are always one job log away -- which is the reason
# they are recorded in exactly one file, this one.)
#
# ----------------------------------------------------------------------------
# CURRENT: 2026-11-05, run 42048036223 (fastfields-lib#147)
#   image  ubuntu-26.04 (runner-images 20261025.179), via `ubuntu-latest`
#   nvcc   12.4.131 (nvidia-cuda-toolkit 12.4.131~12.4.1-8)
#   flags  -O1, -j2, INDEXFLAGS default (FF_INDEX32=1), BOUNDFLAGS/SPLINEFLAGS
#          as set above
# ----------------------------------------------------------------------------
#
#     reg_field_rls        6.07 GiB   568.71 s   <-- the heaviest module now
#     reg_field            5.32 GiB   527.96 s
#     resize               2.08 GiB   195.54 s
#     reg_flow_matvec_3d   1.97 GiB   187.69 s   <-- heaviest reg_flow slice
#     reg_flow_rls         1.84 GiB   158.95 s
#     pushpull_backward    1.50 GiB   177.92 s
#     pushpull             1.48 GiB   169.55 s
#     restrict             1.34 GiB   126.15 s
#     reg_flow_matvec_2d   1.23 GiB    97.05 s
#     reg_flow_diag_3d     1.03 GiB    59.72 s
#     reg_flow_relax_3d    1.00 GiB    99.80 s
#     distance             0.99 GiB   135.40 s
#     reg_flow_kernel_3d   0.86 GiB    75.09 s
#     reg_flow_diag_2d     0.65 GiB    42.23 s
#     reg_flow_relax_2d    0.63 GiB    49.61 s
#     reg_flow_matvec_1d   0.60 GiB    44.97 s
#     reg_flow_kernel_2d   0.55 GiB    44.63 s
#     reg_flow_kernel_1d   0.46 GiB    34.66 s
#     posdef               0.46 GiB    38.21 s
#     splinc               0.43 GiB    68.83 s
#     reg_flow_diag_1d     0.41 GiB    31.43 s
#     reg_flow_relax_1d    0.27 GiB    20.79 s
#     reg_flow (front)     0.20 GiB     1.37 s
#
#   `make cuda -j2` wall  1552 s (25m52s), from the job's own step timings.
#
# INDEX64 IS NOT RE-MEASURED HERE. The `build-cuda` matrix builds both index
# positions, but on that run the FF_INDEX32=0 leg sat in `apt-get install
# nvidia-cuda-toolkit` for 37 minutes (the x64 leg's identical step took 130 s)
# and produced no table -- a mirror stall, not a build result. So the only
# FF_INDEX32=0 figures in this file are the ubuntu-24.04 / nvcc 12.0.140 ones
# further down, and they should be read as that image's. Re-deriving them is
# one green `build-cuda (index64)` away; do not copy the 24.04 column up here
# as though it were current.
#
# WHAT MOVED, AND IT IS NOT UNIFORM. Against the previous recording (same
# flags, same split, ubuntu-24.04 / nvcc 12.0.140, run 32418602148):
#
#     reg_field        8.35 -> 5.32 GiB   -36%
#     reg_field_rls    7.07 -> 6.07 GiB   -14%
#     ...everything else within a few percent: reg_flow_matvec_3d 1.94 -> 1.97,
#     resize 2.00 -> 2.08, reg_flow_rls 1.90 -> 1.84, pushpull_backward
#     1.51 -> 1.50, pushpull 1.48 -> 1.48, restrict 1.30 -> 1.34, distance
#     0.99 -> 0.99, splinc 0.42 -> 0.43, posdef 0.43 -> 0.46, reg_flow front
#     0.20 -> 0.20.
#
# So the new nvcc got materially cheaper on exactly the two modules that are
# NOT split, and changed nothing measurable anywhere else -- including the
# twelve reg_flow slices, which is the useful control. It also reordered the
# table: `reg_field_rls` is now the heaviest thing in this directory, where
# every previous recording had `reg_field` ahead of it. Any statement of the
# form "the ordering is stable" is retired.
#
# THE HEADROOM THIS BUYS IS MARGIN, NOT BUDGET. FF_MEM_BUDGET_KB stays at
# 14 GiB, -j2 stays, -O1 stays, and the MODULES split stays. The ceiling is
# not "does the heaviest module fit" -- it is "do two heavy modules peak at
# once", which no per-module measurement can answer (see below) and which a
# newer nvcc has not made safe, only roomier. A lower measurement is a reason
# to stop worrying, not a reason to spend.
#
# ----------------------------------------------------------------------------
# HISTORY: 2026-08-19, fastfields-lib#80 -- image ubuntu-24.04, nvcc 12.0.140
# ----------------------------------------------------------------------------
# Kept because it is where the "~3.8 GB per split module, ~6-7 GB combined"
# guess was replaced by measurement -- that guess was WRONG, and wrong in the
# dangerous direction. These are NOT current figures:
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
# The `reg_flow` row stopped describing anything compiled when #147 split it
# (below). On that image the heavy modules reproduced run to run to better
# than 0.1% and the small ones wandered a few tenths of a GB (distance
# 0.77 -> 0.99 was the worst) -- which is the run-to-run spread, on one image,
# that the per-module budget has to absorb. The cross-image spread above is an
# order of magnitude larger.
#
# ~~~ AND -j2 IS STILL LUCK, EVEN WITH THE 2026-11 HEADROOM ~~~
# Worth being blunt about, because a lower measurement looks like permission.
# Feed the measured wall times above through make's -j2 dispatch order (which
# is MODULES order) and the heavy modules do NOT take turns -- they overlap.
# On the current image, with reg_flow split, the worst pair left is:
#
#     reg_field + reg_field_rls   overlap ~530 s   peaks sum to 11.39 GiB
#
# which is 71% of a 16 GB runner, and it is the ONLY pair that still matters:
# the twelve reg_flow slices peak at 1.97 GiB and under, so any slice
# overlapping anything is comfortable. On the previous image the same pair
# summed to 15.42-15.59 GB -- i.e. it cleared the limit by a few hundred MB,
# and `reg_field + reg_flow` (unsplit) did not clear it at all, at 21.90 GB.
#
# So this is genuinely better than it was, by two independent changes (the
# #147 split and a cheaper nvcc) -- and it is NOT a licence to raise -j:
#
#   * The 11.39 GiB is a sum of two maxima, not a measured co-peak. A module's
#     peak RSS is a brief spike late in its own ptxas phase; the build is green
#     because two spikes have not landed in the same instant. Nothing enforces
#     that, and no per-module budget can see it.
#   * Anything that shifts the relative phase of two heavy compiles --
#     reordering MODULES, a different nvcc, ccache, a bound/spline policy
#     change, adding -arch/-gencode targets -- reshuffles it, and the failure
#     mode is an OOM kill that reads as a random infrastructure flake rather
#     than a build decision.
#   * The headroom arrived with the runner image and can leave with it. The
#     36% drop on reg_field was given by nvcc 12.4, not earned here, and the
#     next image can take it back; -j4 chosen on today's figures would then
#     OOM on a build nobody changed.
#
# So: do NOT raise -j, do NOT drop -O1, and do NOT read a green build or a
# lower table as proof that two regularisers fit side by side. The `build-cuda`
# job fails if any single module exceeds FF_MEM_BUDGET_KB (14 GiB), which turns
# the approach to the ceiling into a diagnosable error instead of a mystery
# OOM; it does not and cannot catch two peaks coinciding. 14 GiB stays even
# though the current maximum is 6.07 -- the budget is a tripwire for "something
# made a module materially heavier", and tightening it onto today's figures
# would make it fire on the next image instead.
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
# Adding them does not move the ceiling: in the CURRENT table above the four
# are among the cheapest modules in the file (resize 2.08, restrict 1.34,
# posdef 0.46, splinc 0.43 GiB), together well under a single regulariser.
# They cost ~7 min of the ~26 min compile.
# ~~~ reg_flow IS SPLIT ACROSS TRANSLATION UNITS (fastfields-lib#147) ~~~
#
# reg_flow was the 12.98 GiB entry in the HISTORY table above -- 81% of a
# 16 GB runner, in one indivisible TU, and the single number that pinned this
# build at -j2. (It has not been re-measured unsplit on the current image and
# cannot be from this branch: there is no single reg_flow TU any more.)
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
# ~~~ MEASURED, AND RE-MEASURED AFTER fastfields-lib#155 ~~~
#
# Both sides below are post-#155 runs of this same job, so nothing here is
# inherited from the build that #155 fixed. That matters because #155 was
# nvcc replacing the body of every impl/kernels/utils.h helper IN THE HOST
# OBJECT with ::exit(1) -- so the first measurement of this split was taken
# against a library that could not run. It turns out not to have moved the
# compile at all:
#
#     reg_flow, one TU, index32       peak        elapsed
#       pre-#155  (run 32365351378)   12.98 GiB   1149.89 s
#       post-#155 (run 32397332659)   12.98 GiB   1069.30 s
#
# The peak is identical because the code #155 restored is HOST code, and the
# peak belongs to a device-side process (cicc/ptxas). A correctness
# catastrophe and a compile-cost non-event.
#
#   BEFORE  run 32397332659, commit 0d40731 (unsplit)
#   AFTER   run 32418602148, commit 2c4842b (this split)
#   Same image (ubuntu-24.04), nvcc 12.0.140, -O1, same bound/spline policy,
#   -j2.
#
# EVERY NUMBER IN THIS A/B IS AN ubuntu-24.04 / nvcc 12.0.140 NUMBER, and it
# stays that way on purpose: it is a controlled before/after of the split, and
# re-taking one side on a different image would destroy the control rather than
# refresh it. For today's absolute figures use the CURRENT table above. The
# control survives the image move intact -- the twelve slices re-measured on
# ubuntu-26.04 / nvcc 12.4.131 land within a few percent of the AFTER column
# below (heaviest 1.94 -> 1.97 GiB), so the split's case is not an artefact of
# the nvcc it was measured on.
#
# ~~~ index32 (the default, and what ships) ~~~
#
#     reg_flow_matvec_3d  1.94 GiB  177.13 s   reg_flow_matvec_1d  0.61  48.84
#     reg_flow_matvec_2d  1.22 GiB   98.53 s   reg_flow_relax_2d   0.60  48.01
#     reg_flow_diag_3d    1.03 GiB   64.72 s   reg_flow_kernel_2d  0.59  49.71
#     reg_flow_relax_3d   0.96 GiB   89.84 s   reg_flow_kernel_1d  0.48  38.77
#     reg_flow_kernel_3d  0.89 GiB   83.19 s   reg_flow_diag_1d    0.44  35.04
#     reg_flow_diag_2d    0.67 GiB   46.24 s   reg_flow_relax_1d   0.26  21.66
#                                              reg_flow (front)    0.20   1.64
#
#   heaviest reg_flow TU   12.98 -> 1.94 GiB   (-85.0%)
#   reg_flow elapsed, sum  1069.30 -> 803.32 s (-24.9%)
#   `make cuda -j2` wall   2145 -> 1861 s      (-13.2%)
#   max peak, ANY module   12.98 (reg_flow) -> 8.35 (reg_field)
#
#   (On the current image that last line reads 6.07 GiB, reg_field_rls -- the
#   split still removes reg_flow as the constraint, but which module inherits
#   the title changed with nvcc. CURRENT table above.)
#
# All thirteen per-slice peaks reproduce the pre-#155 measurement of the same
# split to within 0.01 GiB, and reproduce OFF-RUNNER too: measured locally on
# nvcc 12.0.140 at the same flags, the front TU came out 205532 kB against
# CI's 205332 (0.1%), reg_flow_diag_1d 456904 against 456160 (0.2%). Peak RSS
# is the reproducible number here. WALL-CLOCK IS NOT: reg_field, which this
# change does not touch, moved 8.23 -> 8.35 GiB and 848.87 -> 861.05 s between
# the same two runs, so read the -13.2% as "clearly faster", not as 13.2%.
#
# ~~~ index64 ~~~
#
#   heaviest reg_flow TU   5.69 -> 1.05 GiB   (-81.5%)
#   reg_flow elapsed, sum  482.78 -> 455.23 s (-5.7%)
#   `make cuda -j2` wall   966 -> 918 s       (-5.0%)
#   max peak, ANY module   5.69 (reg_flow) -> 3.76 (reg_field)
#
# The elapsed saving is far smaller on this leg, and that is the expected
# shape: collapsing the offset axis already removed half the instantiations,
# so there is less left for the split to divide. Memory still falls by 4/5.
#
# THE COUNTER-FORCE IS REAL BUT SMALL, and was measured rather than assumed.
# Every slice re-parses the whole template header set, so total elapsed could
# have risen. reg_flow.cpp IS that measurement: it parses impl/cuda/reg_flow.h
# plus the kernels and instantiates nothing, and costs 0.20 GiB and 1.64 s.
# Twelve extra TUs of re-parsing is ~20 s against a 1069 s baseline -- under
# 2%, an order of magnitude below the instantiation work that now divides.
# That ratio is what makes this split pay, and it is NOT a general law: a
# module whose per-TU instantiation work is comparable to ~1.6 s should not be
# split.
#
# WHAT THIS DOES *NOT* BUY: A HIGHER -j.
# The ceiling is max(peak) over EVERY module, and reg_field and reg_field_rls
# are untouched by this change. reg_flow simply stops being the constraint; a
# reg_field module becomes it, and their overlap is the same -j2 hazard
# described above -- unchanged by this PR, because neither module is touched by
# it. On the measuring image that overlap was 15.42 GiB against a 16 GB runner;
# on the current one it is 11.39 GiB. Neither figure is a licence: see
# "-j2 IS STILL LUCK" above for why a sum of two maxima is not a measured
# co-peak. So do NOT raise -j on the strength of this change, or of the newer
# nvcc. Splitting reg_field and reg_field_rls the same way (they are
# structurally identical files, and the port is a prefix rename of this diff)
# is what would take max(peak) to ~2 GiB and make -j4 CPU-bound instead of
# RAM-bound -- and it would do so by construction rather than by courtesy of
# whichever nvcc the runner image ships.
#
# INTERACTION WITH INDEXFLAGS: they compose, they do not overlap. On the
# heaviest slice, same TU, index axis toggled (the two legs above, so again
# ubuntu-24.04 / nvcc 12.0.140):
#
#     reg_flow_matvec_3d  FF_INDEX32=1  1.94 GiB  177.13 s
#                         FF_INDEX32=0  1.05 GiB  100.01 s
#                                       -45.8%    -43.5%
#
# Splitting does not make the index axis relatively cheaper; it shrinks the
# absolute number the proportion applies to. The two compose -- 12.98 -> 1.94
# GiB from the split, 1.94 -> 1.05 GiB from the knob. So this split is NOT a
# reason to stop caring about FF_INDEX32; what it changes is which resource
# the knob buys back, since memory is no longer what binds this module.
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
