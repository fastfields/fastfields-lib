# libfastfields-cpu.so -- the dtype-dispatch layer.
#
# Never invokes nvcc: `make cpu` must work on a machine with no CUDA toolchain
# at all, which is what keeps a CPU-only build of the whole project possible.

GROUP := lib-cpu
include ../../make/common.mk

CXXFLAGS += -std=c++11 -O3 $(DIAGFLAGS)
TESTFLAGS += $(DIAGFLAGS)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Bound / spline compile policy, PER OPERATION FAMILY
# (see impl/kernels/bounds.h and impl/kernels/spline.h)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Every boundary condition and spline order a module dispatches on is either a
# dedicated static instantiation (fastest per voxel) or routed through the one
# shared `type::Dynamic` instantiation (smaller, cheaper to build). Results are
# identical either way. Which is which is a build-time choice, made here, and
# made separately for each operation family -- each module is its own
# translation unit, so a family is just a set of objects compiled with its own
# flags:
#
#   interpolation  PP_BOUNDFLAGS + PP_SPLINEFLAGS   pushpull, pushpull_backward,
#                                                   resize, restrict, splinc
#   regularisers   REG_BOUNDFLAGS                   reg_field, reg_flow,
#                                                   solve_field
#   (none)                                          distance, posdef
#
# The regularisers have no spline axis, hence no REG_SPLINEFLAGS; add one the
# day a regulariser reads FF_SPLINE_*. `distance` and `posdef` read no policy
# macro at all. Every module must be in exactly one of the three lists (checked
# below), so a new module cannot silently fall back to the header defaults.
#
# Kernel headers never read a policy macro themselves -- only the dispatch .cpp
# files do, choosing template arguments -- so the same template instantiation
# has the same definition in every object whatever its family's flags, and two
# families on different policies cannot violate the ODR.
#
# Precedence, highest first:
#   1. PP_BOUNDFLAGS / PP_SPLINEFLAGS / REG_BOUNDFLAGS given explicitly;
#   2. BOUNDFLAGS / SPLINEFLAGS given on the command line or in the
#      environment -- applied to EVERY family, so existing overrides, the CI
#      matrix and the nightly full-static job keep meaning what they meant
#      (see ff_policy in make/common.mk; an *empty* BOUNDFLAGS="" counts, and
#      means all static);
#   3. the per-family defaults below.
#
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=1" SPLINEFLAGS="-DFF_STATIC_SPLINES=1"
#                                                    # everything static
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0" SPLINEFLAGS="-DFF_STATIC_SPLINES=0"
#                                                    # everything Dynamic
#   make REG_BOUNDFLAGS="-DFF_STATIC_BOUNDS=1"       # one family only
#
# DEFAULTS. The same as src/lib-cuda's, deliberately -- one shipped policy per
# family, whatever the backend -- except for NoCheck (below):
#   interpolation  bounds Zero, DCT2, DST2, DFT static; orders 0-3 static.
#                  Zero, DCT2/DST2 (Neumann/Dirichlet: the symmetric and the
#                  sign-flipping path) and DFT are the conditions interpolation
#                  is used with in practice; Nearest..Cubic are the orders it
#                  is used with, and the four resample has dedicated
#                  closed-form kernels for.
#   regularisers   bounds DCT2, DST2, DFT static -- Neumann, Dirichlet or
#                  periodic is what regularisation uses in practice (see
#                  bounds.h's notes on DCT1/DST1/Replicate self-adjointness).
#   everything else routes through Dynamic.
#
# NoCheck is left EXACTLY as this backend always had it: static here (every
# CPU condition was static before the per-family split), Dynamic on CUDA. That
# is an OPEN QUESTION, not a decision -- whether NoCheck should stay a static
# fast path here or become Dynamic everywhere has not been settled. Do not read
# the asymmetry as intended.
#
# Kept out of CXXFLAGS on purpose, all of them, so that overriding CXXFLAGS on
# the command line (as the g++ CI leg does) does not silently drop the policy.
PP_BOUND_DEFAULT   = -DFF_STATIC_BOUNDS=0 \
                     -DFF_STATIC_BOUND_ZERO=1 -DFF_STATIC_BOUND_DCT2=1 \
                     -DFF_STATIC_BOUND_DST2=1 -DFF_STATIC_BOUND_DFT=1 \
                     -DFF_STATIC_BOUND_NOCHECK=1
PP_SPLINE_DEFAULT  = -DFF_STATIC_SPLINES=0 \
                     -DFF_STATIC_SPLINE_NEAREST=1 -DFF_STATIC_SPLINE_LINEAR=1 \
                     -DFF_STATIC_SPLINE_QUADRATIC=1 -DFF_STATIC_SPLINE_CUBIC=1
REG_BOUND_DEFAULT  = -DFF_STATIC_BOUNDS=0 \
                     -DFF_STATIC_BOUND_DCT2=1 -DFF_STATIC_BOUND_DST2=1 \
                     -DFF_STATIC_BOUND_DFT=1 \
                     -DFF_STATIC_BOUND_NOCHECK=1

PP_BOUNDFLAGS  ?= $(call ff_policy,BOUNDFLAGS,$(PP_BOUND_DEFAULT))
PP_SPLINEFLAGS ?= $(call ff_policy,SPLINEFLAGS,$(PP_SPLINE_DEFAULT))
REG_BOUNDFLAGS ?= $(call ff_policy,BOUNDFLAGS,$(REG_BOUND_DEFAULT))

PP_MODULES       = pushpull pushpull_backward resize restrict splinc
REG_MODULES      = reg_field reg_flow solve_field
NOPOLICY_MODULES = distance posdef

# POLICY_<module>: the policy flags that module's object is compiled with.
$(foreach m,$(PP_MODULES),$(eval POLICY_$(m) = $$(PP_BOUNDFLAGS) $$(PP_SPLINEFLAGS)))
$(foreach m,$(REG_MODULES),$(eval POLICY_$(m) = $$(REG_BOUNDFLAGS)))

# 32-bit index compile policy (see core/dispatch.h) -- the same idea as
# the bound/spline policy, one axis further out again, and the most expensive of
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

# `make test` compiles the suite under exactly the policy the library ships:
# there are no target-specific defaults any more, so shipped == tested by
# construction. (There used to be a sparser `test:` default, from the days when
# `-DFF_TEST_SPARSE` made pushpull/resize/restrict *throw* on uncovered
# combinations. Dynamic routing made that redundant -- every combination stays
# functional -- and the library default is now itself a mixed policy.)
# Overrides work exactly as for the library: `make test BOUNDFLAGS=...
# SPLINEFLAGS=...` (the CI matrix's legs) applies to both families, PP_* /
# REG_* to one.

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

_UNCLASSIFIED = $(filter-out $(PP_MODULES) $(REG_MODULES) $(NOPOLICY_MODULES),$(MODULES))
ifneq ($(strip $(_UNCLASSIFIED)),)
  $(error module(s) "$(_UNCLASSIFIED)" not assigned to a bound/spline policy family: add to PP_MODULES, REG_MODULES or NOPOLICY_MODULES)
endif

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
	$(CXX) $(CXXFLAGS) $(POLICY_$*) $(INDEXFLAGS) $(INCLUDES) $(PICFLAG) -MMD -MP -c -o $@ $<

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
# Test module objects get the same per-family policy flags as the library's
# (POLICY_<module>, above). They still live in their own dir: a `make test`
# with a policy override must never leave objects behind that a later `make
# all` would link into the shipped library. The test drivers read no policy
# macro and get none.

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
TESTCPPFLAGS = $(CXXFLAGS) $(INDEXFLAGS) $(INCLUDES) -MMD -MP

$(TESTOBJDIR):
	$(MKDIR) $(TESTOBJDIR)

$(TESTOBJDIR)/%.$(MOSUF): %.cpp | $(TESTOBJDIR)
	$(CXX) $(TESTCPPFLAGS) $(POLICY_$*) -c -o $@ $<

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
