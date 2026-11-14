# libfastfields.so -- the device-dispatch hub.
#
# Links against libfastfields-cpu.so always, and libfastfields-cuda.so when
# USE_CUDA=1.

GROUP := lib
include ../../make/common.mk

CXXFLAGS += $(CXXSTD) -O3 $(DIAGFLAGS)
TESTFLAGS += $(DIAGFLAGS)

# Build the CUDA backend and link it in (needs nvcc). Default off: the CPU path
# is the tested source of truth and CI has no GPU.
USE_CUDA ?= 0
ifeq ($(USE_CUDA),1)
  CXXFLAGS    += -DFF_WITH_CUDA
  CUDA_LDFLAGS = $(call LINKLIB,$(LIBDIR),fastfields-cuda)
  CUDA_DEP     = $(LIBDIR)/libfastfields-cuda.$(SOSUF)
endif

MODULES = \
	distance \
	posdef \
	resize \
	restrict \
	splinc \
	pushpull \
	reg_field \
	reg_flow \
	solve_field

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
#
# $(NO_UNDEFINED) (-Wl,--no-undefined on GNU ld/lld; empty where the platform
# linker has no such option) makes this link the gate on backend completeness.
# The hub dispatches to FF_CPU:: and, under FF_WITH_CUDA, FF_CUDA:: symbols it
# only declares; a shared object may carry unresolved symbols, so without this
# flag a backend that simply never compiled the module defining one of them
# links clean and fails on a GPU instead. That is exactly fastfields-lib#80:
# src/lib-cuda's MODULES omitted posdef/resize/restrict/splinc, leaving eleven
# FF_CUDA:: entry points undefined with every build green. The flag applies to
# the CPU-only link too -- libfastfields-cpu.so is complete, so it costs
# nothing and keeps the same guarantee on the leg CI runs on every push.
$(TARGET): $(OBJECTS) $(CPU_DEP) $(CUDA_DEP) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(PLATFORMFLAGS) $(PLATFORMLDFLAGS) -shared $(PICFLAG) $(SONAME_FLAG) $(RPATH) \
	  $(NO_UNDEFINED) $(call LINKLIB,$(LIBDIR),fastfields-cpu) $(CUDA_LDFLAGS) \
	  -o $@ $(OBJECTS)

# -MMD -MP emit header dependency files (*.d). This layer had none, which in a
# header-only codebase meant editing a kernel header left a stale object.
#
# -DFF_BUILDING_LIB: these objects define the FF_API entry points (exported
# from the DLL on Windows) and *import* the FF_CPU_API ones from
# libfastfields-cpu -- see core/export.h.
$(OBJDIR)/%.$(MOSUF): %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(PLATFORMFLAGS) -DFF_BUILDING_LIB $(INCLUDES) $(PICFLAG) -MMD -MP -c -o $@ $<

########################################################################
#	Tests
########################################################################
# Standalone, header-only tests: each tests/lib/test_<x>.cpp includes only the
# hub headers it exercises (checks.h, splinc.h, ...), so it compiles and runs on
# its own without linking libfastfields.so. Op correctness stays gated by
# lib-cpu's suite -- these cover the argument validation that lives in this
# layer and nowhere below it.

TESTSRCDIR = $(ROOTDIR)/tests/lib
TESTSRC    = $(wildcard $(TESTSRCDIR)/test_*.cpp)
TESTBINS   = $(patsubst $(TESTSRCDIR)/%.cpp,$(TESTDIR)/%$(EXESUF),$(TESTSRC))

test: $(TESTBINS)
	$(call verb, "Running tests...")
	@ for t in $(TESTBINS); do echo "--- $$t"; $$t || exit 1; done
	$(call verb, "Running tests: done.")

$(TESTDIR)/test_%$(EXESUF): $(TESTSRCDIR)/test_%.cpp | $(TESTDIR)
	$(CXX) $(CXXFLAGS) $(PLATFORMFLAGS) $(PLATFORMLDFLAGS) $(TESTFLAGS) -DFF_STATIC_LINK $(INCLUDES) -MMD -MP -o $@ $<

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
