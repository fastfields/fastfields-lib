# libfastfields.so -- the device-dispatch hub.
#
# Links against libfastfields-cpu.so always, and libfastfields-cuda.so when
# USE_CUDA=1.

GROUP := lib
include ../../make/common.mk

CXXFLAGS += -std=$(STD) -O3 $(DIAGFLAGS)
TESTFLAGS += $(DIAGFLAGS)

# Build the CUDA backend and link it in (needs nvcc). Default off: the CPU path
# is the tested source of truth and CI has no GPU.
USE_CUDA ?= 0
ifeq ($(USE_CUDA),1)
  CXXFLAGS    += -DFF_WITH_CUDA
  CUDA_LDFLAGS = -L$(LIBDIR) -lfastfields-cuda
  CUDA_DEP     = $(LIBDIR)/libfastfields-cuda.$(SOSUF)
endif

# NB: this is `teeny`'s module set. `main` additionally builds
# solve_field -- those sources are present in this tree (teeny descends from the
# consolidated main) but teeny's kernels do not implement them yet. That
# catch-up is tracked work, deliberately not done as part of the merge.
MODULES = \
	distance \
	posdef \
	resize \
	restrict \
	splinc \
	pushpull \
	reg_field \
	reg_flow

OBJECTS = $(addprefix $(OBJDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
TARGET  = $(BUILDDIR)/libfastfields.$(SOSUF)
CPU_DEP = $(LIBDIR)/libfastfields-cpu.$(SOSUF)

.PHONY: all install test clean clean-obj clean-lib

all: verb.build.lib $(TARGET) verb.build.lib.done

install: all

$(CPU_DEP):
	$(MAKE) -C ../lib-cpu all

$(LIBDIR)/libfastfields-cuda.$(SOSUF):
	$(MAKE) -C ../lib-cuda all

# Real prerequisites (not just siblings under a recipe-less target) so `make -j`
# cannot run this link step before the backend libraries it links against
# actually exist. ($^ would pull the .so prerequisites into the link line too,
# so the recipe lists $(OBJECTS) explicitly instead.)
$(TARGET): $(OBJECTS) $(CPU_DEP) $(CUDA_DEP) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -shared $(PICFLAG) $(SONAME_FLAG) $(RPATH) \
	  -L$(LIBDIR) -lfastfields-cpu $(CUDA_LDFLAGS) \
	  -o $@ $(OBJECTS)

# -MMD -MP emit header dependency files (*.d). This layer had none, which in a
# header-only codebase meant editing a kernel header left a stale object.
$(OBJDIR)/%.$(MOSUF): %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PICFLAG) -MMD -MP -c -o $@ $<

########################################################################
#	Tests
########################################################################
# Standalone, header-only tests: each tests/lib/test_<x>.cpp includes only the
# hub headers it exercises (checks.h, splinc.h, ...), so it compiles and runs on
# its own without linking libfastfields.so. Op correctness stays gated by
# lib-cpu's suite -- these cover the argument validation that lives in this
# layer and nowhere below it.

TESTSRCDIR = $(ROOTDIR)/tests/lib
# See the note on TESTEXCLUDE in src/lib-cpu/Makefile: test_splinc_bound.cpp
# exercises ff::require_splinc_bound, which exists on `main` but not on this
# branch. Excluded rather than deleted so the gap stays visible.
TESTEXCLUDE = test_splinc_bound.cpp

TESTSRC    = $(filter-out $(addprefix $(TESTSRCDIR)/,$(TESTEXCLUDE)),\
                $(wildcard $(TESTSRCDIR)/test_*.cpp))
TESTBINS   = $(patsubst $(TESTSRCDIR)/%.cpp,$(TESTDIR)/%,$(TESTSRC))

test: $(TESTBINS)
	$(call verb, "Running tests...")
	@ for t in $(TESTBINS); do echo "--- $$t"; $$t || exit 1; done
	$(call verb, "Running tests: done.")

$(TESTDIR)/test_%: $(TESTSRCDIR)/test_%.cpp | $(TESTDIR)
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) $(INCLUDES) -MMD -MP -o $@ $<

-include $(wildcard $(OBJDIR)/*.d)
-include $(wildcard $(TESTDIR)/*.d)

clean: clean-obj clean-lib
clean-obj:
	$(DEL) -r $(OBJDIR)
clean-lib:
	$(DEL) $(TARGET)

verb.build.lib:
	$(call verb, "Building library...")
verb.build.lib.done:
	$(call verb, "Building library: done.")
