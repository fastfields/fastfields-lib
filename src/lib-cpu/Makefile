define verb
	@ echo "_____________________________________________________________"
	@ echo ""
	@ echo "        " $(1)
	@ echo "_____________________________________________________________"
	@ echo ""
endef

########################################################################
# 	Optional
########################################################################

COPY        ?= cp -f
DEL         ?= rm -f
MOVE        ?= mv -f
MKDIR     	?= mkdir -p
BUILDDIR  	?= ./build
CXXFLAGS  	+= -std=c++17 -O3 -ferror-limit=1 -ftemplate-backtrace-limit=0
# teeny's anyrank inline meta store caps at TNY_MAX_RANK; bump to DLPack's max
# (64) so a deep-batch pushpull tensor is not assert-aborted (see pushpull.cpp).
CXXFLAGS  	+= -DTNY_MAX_RANK=64
# teeny (header-only) + its vendored CCCL, reached through the impl/kernels nesting.
TEENYDIR  	?= impl/kernels/external/teeny
INCLUDES  	+= -I$(TEENYDIR)/include -I$(TEENYDIR)/external/cccl/libcudacxx/include
TESTFLAGS 	+= -ferror-limit=1 -ftemplate-backtrace-limit=0
UNAME     	?= uname
GET_ARCH  	?= $(UNAME) -m
MOSUF 	  	 = o
SOSUF      	 = so
SONAME     	 = soname
OMPFLAG    	 = -fopenmp
USE_OPENMP 	?= 0
# Position-independent code + shared-library soname flags. Both are POSIX-only:
# on Windows the PE/COFF linker has no soname and code is position-independent
# by default, so they are cleared in the Windows block below and referenced via
# these variables (never hard-coded) in the link rules.
PICFLAG       = -fPIC
SONAME_PREFIX = -Wl,-$(SONAME),
# Full soname linker flag for the shared lib, empty on Windows (see below):
# e.g. -Wl,-soname,libfastfields-cpu.so. $(@F) is the target's file name,
# expanded at link time (SONAME_FLAG is recursively expanded).
SONAME_FLAG   = $(if $(SONAME_PREFIX),$(SONAME_PREFIX)$(@F))

########################################################################
# 	Platform-specific settings
########################################################################

# Native Windows (cmd/pwsh) has no `uname`; GNU make sets OS=Windows_NT there.
# Under a Unix-like shell on Windows (git bash / MSYS, which is what CI uses),
# `uname` returns MINGW*/MSYS and the block further down matches instead.
ifeq ($(OS),Windows_NT)
  PLATFORM   = Windows
endif
ifndef PLATFORM
  PLATFORM   = $(shell $(UNAME))
  ifeq (Darwin,$(PLATFORM))
    ifeq (arm64,$(shell $(GET_ARCH))) # Check for Apple Silicon
      PLATFORM = arm64
    endif
  endif
endif

##### macOS #####
ifeq (Darwin,$(PLATFORM))
  OMPFLAG    = -fopenmp=libiomp5
  SOSUF      = dylib
  SONAME     = install_name
endif
ifeq (arm64,$(PLATFORM))
  OMPFLAG    = -fopenmp=libiomp5
  SOSUF      = dylib
  SONAME     = install_name
endif

##### Windows (native OS=Windows_NT, or a Unix-like shell: MINGW*/MSYS) #####
# Built with clang++ (as CI does): objects stay .o, shared libs are .dll, and
# there is no soname / -fPIC (the PE/COFF linker rejects -Wl,-soname and code is
# position-independent by default). A Unix-like shell (git bash / MSYS) is
# required so the recipe commands (cp/rm/mkdir/uname) resolve.
IS_WINDOWS =
ifeq (Windows,$(PLATFORM))
  IS_WINDOWS = 1
endif
ifeq (MINGW32,$(word 1,$(subst _, ,$(PLATFORM))))
  IS_WINDOWS = 1
endif
ifeq (MINGW64,$(word 1,$(subst _, ,$(PLATFORM))))
  IS_WINDOWS = 1
endif
ifeq (MSYS,$(word 1,$(subst _, ,$(PLATFORM))))
  IS_WINDOWS = 1
endif
ifdef IS_WINDOWS
  SOSUF         = dll
  MOSUF         = o
  PICFLAG       =
  SONAME_PREFIX =
endif

########################################################################
# 	Public Targets
########################################################################

all: libcpu

install: libcpu | $(PREFIX)/lib
	$(COPY) $(BUILDDIR)/libfastfields-cpu.$(SOSUF) $(PREFIX)/lib

clean: clean-lib clean-obj

.PHONY: all clean

########################################################################
# 	Build directory
########################################################################

$(BUILDDIR):
	$(MKDIR) $(BUILDDIR)

$(PREFIX)/lib:
	$(MKDIR) $(PREFIX)/lib

########################################################################
# 	Clean
########################################################################

clean-obj:
	$(DEL) $(BUILDDIR)/*.$(MOSUF)

clean-lib:
	$(DEL) $(BUILDDIR)/*.$(SOSUF)

########################################################################
# 	Library
########################################################################

MODULES = \
	distance \
	posdef \
	resize \
	restrict \
	splinc \
	pushpull \
	reg_field \
	reg_flow

OBJECTS  = $(addprefix $(BUILDDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
CPPFILES = $(addsuffix .cpp,$(MODULES))

libcpu: \
	verb.build.lib \
	$(BUILDDIR)/libfastfields-cpu.$(SOSUF) \
	verb.build.lib.done

$(BUILDDIR)/libfastfields-cpu.$(SOSUF): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared $(PICFLAG) $(SONAME_FLAG) -o $@ $^

########################################################################
# 	Objects
########################################################################

# -MMD -MP emit header dependency files (*.d) so that editing a kernel/impl
# header (this is a header-only codebase) rebuilds the affected library object
# instead of leaving a stale binary.
$(BUILDDIR)/%.$(MOSUF): %.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PICFLAG) -MMD -MP -c -o $@ $<

########################################################################
# 	Tests
########################################################################
# Each tests/test_<name>.cpp is linked against the module objects and run.
# Usage: `make test CXX=clang++`.
#
# Every module .cpp is compiled to a shared object ONCE (in build/testobj/)
# and linked into every test binary, instead of being recompiled together
# with each test. This turns ~(#tests x #modules) module compiles into just
# #modules, and the single-source `-c` compiles let ccache cache each object.
#
# Test objects build with -DFF_TEST_SPARSE: the heavy order x bound modules
# (pushpull, resize, restrict, splinc) then instantiate only a covering subset
# of the matrix, cutting test-compile time. The library build (`make all`) omits
# the flag and compiles the full matrix, so it is also the compile gate. The
# test objects live in their own dir so they never collide with the library
# objects built without FF_TEST_SPARSE.

TESTOBJDIR = $(BUILDDIR)/testobj
TESTSRC    = $(wildcard tests/test_*.cpp)
TESTBIN    = $(patsubst tests/%.cpp,$(BUILDDIR)/%,$(TESTSRC))

# Shared per-module objects: compiled once, linked into every test binary.
TESTMODOBJ = $(addprefix $(TESTOBJDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
# Per-test driver objects (one per tests/test_*.cpp).
TESTDRVOBJ = $(patsubst tests/%.cpp,$(TESTOBJDIR)/%.$(MOSUF),$(TESTSRC))

# These objects are built via pattern rules, so make would treat them as
# intermediate and delete them after linking — recompiling everything on the
# next `make test`. Mark them SECONDARY so they persist (real incremental
# rebuilds + warm ccache).
.SECONDARY: $(TESTMODOBJ) $(TESTDRVOBJ)

# -MMD -MP emit header dependency files (*.d) so header edits trigger rebuilds.
TESTCPPFLAGS = $(CXXFLAGS) -DFF_TEST_SPARSE $(INCLUDES) -I. -MMD -MP

$(TESTOBJDIR):
	$(MKDIR) $(TESTOBJDIR)

# Module object (built once, shared across all tests).
$(TESTOBJDIR)/%.$(MOSUF): %.cpp | $(TESTOBJDIR)
	$(CXX) $(TESTCPPFLAGS) -c -o $@ $<

# Test-driver object.
$(TESTOBJDIR)/test_%.$(MOSUF): tests/test_%.cpp | $(TESTOBJDIR)
	$(CXX) $(TESTCPPFLAGS) -c -o $@ $<

# Link each test binary from its driver object + ALL shared module objects.
# Linking the full module set means cross-module symbol references (e.g.
# test_restrict -> resample in resize.cpp) resolve automatically.
$(BUILDDIR)/test_%: $(TESTOBJDIR)/test_%.$(MOSUF) $(TESTMODOBJ) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TESTBIN)
	@ status=0; for t in $(TESTBIN); do \
	    echo "running $$t"; $$t || status=1; \
	done; exit $$status

# Pull in generated header-dependency files (*.d), if any exist yet. The
# library objects live directly in $(BUILDDIR); the test objects are one level
# down in $(TESTOBJDIR) -- the BUILDDIR wildcard does not recurse, so the two
# sets never overlap.
-include $(wildcard $(BUILDDIR)/*.d)
-include $(wildcard $(TESTOBJDIR)/*.d)

.PHONY: test

########################################################################
# 	Messages
########################################################################

verb.build.lib:
	$(call verb, "Building CPU library...")

verb.build.lib.done:
	$(call verb, "Building CPU library: done.")
