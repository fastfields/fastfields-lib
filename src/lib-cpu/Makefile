# libfastfields-cpu.so -- the dtype-dispatch layer.
#
# Never invokes nvcc: `make cpu` must work on a machine with no CUDA toolchain
# at all, which is what keeps a CPU-only build of the whole project possible.

GROUP := lib-cpu
include ../../make/common.mk

CXXFLAGS += -std=c++11 -O3 $(DIAGFLAGS)
TESTFLAGS += $(DIAGFLAGS)

# Boundary-condition compile policy (see impl/kernels/bounds.h). The CPU backend
# compiles the full static matrix comfortably, so every boundary condition keeps
# its own (fastest) instantiation by default. Set e.g.
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DCT2=1"
# to route the rarely used conditions through the shared `bound::type::Dynamic`
# runtime implementation instead -- smaller library, faster build, identical
# results. Deliberately a separate variable from CXXFLAGS so that overriding
# CXXFLAGS on the command line does not silently drop the policy.
BOUNDFLAGS  ?=

# Interpolation-order compile policy (see impl/kernels/spline.h) -- the same
# idea one axis further out, and for the same reason: `pushpull` templates on
# the spline order as well as the boundary condition, so all eight orders static
# means an 8x8 matrix of instantiations. The CPU backend compiles that
# comfortably, so every order keeps its own (fastest) instantiation by default.
# Set e.g.
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=0 -DFF_STATIC_SPLINE_LINEAR=1"
# to route the remaining orders through the shared `spline::type::Dynamic`
# runtime implementation instead -- identical results, smaller/faster build.
# Kept out of CXXFLAGS for the same reason as BOUNDFLAGS.
SPLINEFLAGS ?=

# 32-bit index compile policy (see core/dispatch.h) -- the same idea as
# BOUNDFLAGS/SPLINEFLAGS, one axis further out again, and the most expensive of
# the three: EVERY templated kernel below this layer is templated on `offset_t`,
# whose two values (int32_t / int64_t) are chosen per call by
# canUse32BitIndexMath. So the narrow path costs exactly x2 instantiations of
# everything, measured at -50.3% instantiations / -44.6% peak compiler RSS on
# reg_flow when dropped (fastfields-lib#94).
#
#   make INDEXFLAGS="-DFF_INDEX32=0"     # one offset type (int64_t) everywhere
#
# Results are identical either way; code size, compile cost and per-voxel speed
# move. The narrow path is an inherited register-pressure optimisation that has
# never been benchmarked in this project, and it matters least here: the CPU
# backend is not the one whose build is memory-constrained, and it has 64-bit
# integer units. It stays ON by default all the same -- flipping it is a
# performance decision and wants a benchmark, not a build tidy-up.
#
# This is a SEPARATE default from src/lib-cuda/Makefile's, which is the whole
# point of putting it here rather than in make/common.mk: the two backends can
# take the axis independently. Kept out of CXXFLAGS for the same reason as
# BOUNDFLAGS/SPLINEFLAGS -- a CXXFLAGS= override must not silently drop policy.
INDEXFLAGS  ?= -DFF_INDEX32=1

# `test` target only: a sparser *default* BOUNDFLAGS/SPLINEFLAGS than the
# library's (empty = fully static). pushpull's own dispatch used to carry a
# second, hand-duplicated switch behind `-DFF_TEST_SPARSE` purely to keep a
# bare `make test` fast (a *covering* subset of the order x bound matrix --
# all bounds for Linear/Cubic, DCT2 only for the rest -- that threw on
# anything else). That is now redundant with, and strictly weaker than,
# routing through `bound::type::Dynamic`/`spline::type::Dynamic`: it is the
# same compile-cost win (fewer static instantiations), but every combination
# stays fully *functional* via the shared Dynamic instantiation rather than
# throwing -- so pushpull.cpp no longer branches on FF_TEST_SPARSE at all;
# there is exactly one PP_ORDER/PP_BOUND, and "sparse vs. full" is purely this
# target-specific default. It matches the `cuda-default` CI matrix leg, so
# `make test` (no override) exercises the same mixed policy CUDA ships.
# An explicit `make test BOUNDFLAGS=... SPLINEFLAGS=...` (as the CI matrix's
# three legs do) always overrides this default -- GNU Make command-line
# variables outrank both plain and target-specific `?=`.
# NB: plain `=`, not `?=` -- BOUNDFLAGS/SPLINEFLAGS are already `?=`-defaulted
# (to empty) above, which "sets" them at parse time, so a target-specific
# `?=` here would see them as already-set and never fire. Plain `=` still
# yields to an explicit command-line override (`make test BOUNDFLAGS=...`):
# command-line-origin variables outrank *any* in-makefile assignment, target-
# specific or not, unless the makefile uses `override` (which this does not).
test: BOUNDFLAGS  = -DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DCT2=1 -DFF_STATIC_BOUND_DST2=1
test: SPLINEFLAGS = -DFF_STATIC_SPLINES=0 -DFF_STATIC_SPLINE_NEAREST=1 \
                    -DFF_STATIC_SPLINE_LINEAR=1 -DFF_STATIC_SPLINE_QUADRATIC=1 \
                    -DFF_STATIC_SPLINE_CUBIC=1

MODULES = \
	distance \
	posdef \
	resize \
	restrict \
	splinc \
	pushpull \
	pushpull_backward \
	reg_field \
	reg_flow \
	solve_field

OBJECTS = $(addprefix $(OBJDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
TARGET  = $(LIBDIR)/libfastfields-cpu.$(SOSUF)

.PHONY: all install test clean clean-obj clean-lib

all: verb.build.lib $(TARGET) verb.build.lib.done

# Kept as a target for compatibility: the library is already built into
# build/lib/, so there is nothing left to install.
install: all

$(TARGET): $(OBJECTS) | $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared $(PICFLAG) $(SONAME_FLAG) -o $@ $^

# -MMD -MP emit header dependency files (*.d) so that editing a kernel/impl
# header (this is a header-only codebase) rebuilds the affected library object
# instead of leaving a stale binary.
$(OBJDIR)/%.$(MOSUF): %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) $(INDEXFLAGS) $(INCLUDES) $(PICFLAG) -MMD -MP -c -o $@ $<

########################################################################
#	Tests
########################################################################
# Each tests/lib-cpu/test_<name>.cpp is linked against the module objects and
# run. Usage: `make test CXX=clang++` (or CXX=g++).
#
# Every module .cpp is compiled to an object ONCE (in build/obj/lib-cpu/test/)
# and linked into every test binary, instead of being recompiled together with
# each test. This turns ~(#tests x #modules) module compiles into just
# #modules, and the single-source `-c` compiles let ccache cache each object.
#
# Test objects build with -DFF_TEST_SPARSE: the heavy order x bound modules
# (pushpull, resize, restrict, splinc) then instantiate only a covering subset
# of the matrix, cutting test-compile time. The library build (`make all`) omits
# the flag and compiles the full matrix, so it is also the compile gate. The
# test objects live in their own dir so they never collide with the library
# objects built without FF_TEST_SPARSE.

TESTSRCDIR = $(ROOTDIR)/tests/lib-cpu
TESTOBJDIR = $(OBJDIR)/test
TESTSRC    = $(wildcard $(TESTSRCDIR)/test_*.cpp)
TESTBIN    = $(patsubst $(TESTSRCDIR)/%.cpp,$(TESTDIR)/%,$(TESTSRC))

TESTMODOBJ = $(addprefix $(TESTOBJDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
TESTDRVOBJ = $(patsubst $(TESTSRCDIR)/%.cpp,$(TESTOBJDIR)/%.$(MOSUF),$(TESTSRC))

# These objects are built via pattern rules, so make would treat them as
# intermediate and delete them after linking -- recompiling everything on the
# next `make test`. Mark them SECONDARY so they persist (real incremental
# rebuilds + warm ccache).
.SECONDARY: $(TESTMODOBJ) $(TESTDRVOBJ)

# -MMD -MP emit header dependency files (*.d) so header edits trigger rebuilds.
TESTCPPFLAGS = $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) $(INDEXFLAGS) -DFF_TEST_SPARSE $(INCLUDES) -MMD -MP

$(TESTOBJDIR):
	$(MKDIR) $(TESTOBJDIR)

$(TESTOBJDIR)/%.$(MOSUF): %.cpp | $(TESTOBJDIR)
	$(CXX) $(TESTCPPFLAGS) -c -o $@ $<

$(TESTOBJDIR)/test_%.$(MOSUF): $(TESTSRCDIR)/test_%.cpp | $(TESTOBJDIR)
	$(CXX) $(TESTCPPFLAGS) -c -o $@ $<

# Link each test binary from its driver object + ALL shared module objects.
# Linking the full module set means cross-module symbol references (e.g.
# test_restrict -> resample in resize.cpp) resolve automatically.
$(TESTDIR)/test_%: $(TESTOBJDIR)/test_%.$(MOSUF) $(TESTMODOBJ) | $(TESTDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TESTBIN)
	@ status=0; for t in $(TESTBIN); do \
	    echo "running $$t"; $$t || status=1; \
	done; exit $$status

-include $(wildcard $(OBJDIR)/*.d)
-include $(wildcard $(TESTOBJDIR)/*.d)

clean: clean-obj clean-lib
clean-obj:
	$(DEL) -r $(OBJDIR)
clean-lib:
	$(DEL) $(TARGET)

verb.build.lib:
	$(call verb, "Building CPU library...")
verb.build.lib.done:
	$(call verb, "Building CPU library: done.")
