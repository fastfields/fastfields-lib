# Shared makefile fragment for the fastfields build.
#
# Included by the root Makefile and by each of src/lib/, src/lib-cpu/ and
# src/lib-cuda/. It owns everything the three groups must agree on: where the
# build output goes, how the platform is detected, and which diagnostic flags
# the detected compiler actually understands.
#
# A group sets GROUP before including this file:
#     GROUP := lib-cpu
#     include ../../make/common.mk

define verb
	@ echo "_____________________________________________________________"
	@ echo ""
	@ echo "        " $(1)
	@ echo "_____________________________________________________________"
	@ echo ""
endef

COMMON_MK := $(lastword $(MAKEFILE_LIST))
ROOTDIR   := $(patsubst %/,%,$(dir $(abspath $(COMMON_MK))))/..
ROOTDIR   := $(abspath $(ROOTDIR))

COPY      ?= cp -f
DEL       ?= rm -f
MOVE      ?= mv -f
MKDIR     ?= mkdir -p
UNAME     ?= uname
GET_ARCH  ?= $(UNAME) -m
NVCC      ?= nvcc

# Output paths. build/ and build/lib/ are load-bearing: fastfields-dlpack's
# setup.py hardcodes both, so libfastfields.so must land in build/ and the
# backend libraries in build/lib/ regardless of which subdirectory's Makefile
# produced them. Objects go under build/obj/<group>/ so the three groups'
# same-named objects (distance.o, posdef.o, ...) cannot collide now that they
# share one build tree.
BUILDDIR  ?= $(ROOTDIR)/build
LIBDIR    ?= $(BUILDDIR)/lib
OBJDIR    ?= $(BUILDDIR)/obj/$(GROUP)
TESTDIR   ?= $(BUILDDIR)/test/$(GROUP)

INCLUDES  += -I$(ROOTDIR)/include

MOSUF      = o
SOSUF      = so
EXESUF     =
SONAME     = soname
OMPFLAG    = -fopenmp
USE_OPENMP ?= 0

# The language standard every C++ layer above the CUDA one compiles to. Set
# per platform (the Windows block raises it to C++14) rather than spelled
# `-std=c++11` in each Makefile, so the exception lives in one place.
CXXSTD     = -std=c++11

# Platform flags every compile AND link line carries: today only macOS'
# ARCHFLAGS (see the macOS block). Like BOUNDFLAGS/SPLINEFLAGS/INDEXFLAGS this
# is deliberately outside CXXFLAGS, so `make CXXFLAGS=...` -- which CI does on
# most legs -- cannot silently drop it.
PLATFORMFLAGS =

# Position-independent code + shared-library soname flags. Both are POSIX-only:
# on Windows the PE/COFF linker has no soname and code is position-independent
# by default, so they are cleared in the Windows block below and referenced via
# these variables (never hard-coded) in the link rules.
PICFLAG       = -fPIC
SONAME_PREFIX = -Wl,-$(SONAME),
SONAME_FLAG   = $(if $(SONAME_PREFIX),$(SONAME_PREFIX)$(@F))
RPATH         = -Wl,-rpath,'$$ORIGIN'/../lib

# Refuse to produce a shared object with unresolved symbols. A shared library is
# allowed to have them by default, which is how fastfields-lib#80 stayed green:
# four modules were missing from src/lib-cuda's MODULES, so eleven FF_CUDA::
# entry points the hub calls unconditionally were simply absent from
# libfastfields-cuda.so, and neither the CUDA compile nor the hub link said a
# word -- the failure was deferred to a GPU load that CI never performs. With
# this flag that class of omission is a link error.
#
# GNU ld / lld spelling. ld64 (macOS) rejects --no-undefined and already
# defaults to erroring for a dylib; the PE/COFF linker has no equivalent and
# likewise cannot leave symbols unresolved. So both non-Linux blocks below clear
# it, and link rules must reference the variable rather than the flag.
NO_UNDEFINED  = -Wl,--no-undefined

########################################################################
#	Compiler detection
########################################################################

# The diagnostic flags are spelled differently by the two compilers, and each
# rejects the other's outright -- so `make CXX=g++` used to fail on the *flags*
# before it reached a single line of source, and only worked via a command-line
# CXXFLAGS= override that dropped them. Detect instead.
CXX_VERSION := $(shell $(CXX) --version 2>/dev/null | head -1)
ifneq (,$(findstring clang,$(CXX_VERSION)))
  CXX_IS_CLANG := 1
endif

ifdef CXX_IS_CLANG
  DIAGFLAGS ?= -ferror-limit=1 -ftemplate-backtrace-limit=0
else
  DIAGFLAGS ?= -fmax-errors=1 -ftemplate-backtrace-limit=0
endif

########################################################################
#	Platform-specific settings
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
# `Darwin` is an Intel host and `arm64` an Apple Silicon one (see above); the
# build is identical on both, so the two are folded into IS_DARWIN here.
#
# Install names are `@rpath/<file>`, not the bare file name. The install name
# is what every *consumer* of a dylib records as the path to load it from, so
# a bare `libfastfields-cpu.dylib` made the hub look for it in the process's
# working directory. With `@rpath/` the hub resolves it through its own rpath
# (@loader_path/../lib, i.e. build/lib/ beside build/libfastfields.dylib) --
# the Mach-O spelling of the $ORIGIN arrangement Linux uses -- and
# fastfields-dlpack's extension resolves libfastfields.dylib through *its*
# rpath (@loader_path/lib) in turn.
#
# ARCHFLAGS: the target architecture, e.g. `-arch arm64` or `-arch x86_64`.
# cibuildwheel exports exactly this variable when it builds a macOS wheel, and
# a plain `make` used to ignore it, so a wheel labelled for one architecture
# could carry libraries for the other. Empty means the compiler's default (the
# host). It travels in PLATFORMFLAGS (below), not CXXFLAGS, so a CXXFLAGS=
# override cannot drop it. MACOSX_DEPLOYMENT_TARGET, the other half of a wheel's
# platform tag, is read by the compiler straight from the environment.
IS_DARWIN =
ifeq (Darwin,$(PLATFORM))
  IS_DARWIN = 1
endif
ifeq (arm64,$(PLATFORM))
  IS_DARWIN = 1
endif
ifdef IS_DARWIN
  OMPFLAG       = -fopenmp=libiomp5
  SOSUF         = dylib
  SONAME        = install_name
  SONAME_PREFIX = -Wl,-install_name,@rpath/
  RPATH         = -Wl,-rpath,@loader_path/../lib
  NO_UNDEFINED  =
  ARCHFLAGS    ?=
  PLATFORMFLAGS = $(ARCHFLAGS)
endif

##### Windows (native OS=Windows_NT, or a Unix-like shell: MINGW*/MSYS) #####
# Built with LLVM's GNU-style `clang++` driver targeting x86_64-pc-windows-msvc
# (what CI uses): the MSVC STL and the UCRT, the same C runtime CPython is
# built against. Objects stay .o, shared libraries are .dll, and there is no
# soname / -fPIC / rpath (the PE/COFF linker rejects -Wl,-soname and code is
# position-independent by default). A POSIX `sh` on PATH (Git for Windows')
# is required so the recipe commands (cp/rm/mkdir) resolve; CI runs a native
# GNU make, whose $(abspath) yields drive-letter paths the compiler accepts.
#
# Linking a .dll also writes its import library, `<name>.lib`, beside it --
# that is what a consumer links against (a PE/COFF linker cannot link a .dll
# directly), so it is exposed as IMPLIBSUF. Only symbols marked FF_API /
# FF_CPU_API (core/export.h) are exported; a DLL exports nothing by default.
#
# -std=c++14, not the C++11 used everywhere else: the MSVC STL does not support
# C++11 mode. The code itself stays C++11 -- every other platform builds it so.
#
# This block used to be triplicated across the three repos and had already
# drifted -- cuda-lib's copy knew nothing about IS_WINDOWS or PICFLAG and set
# MOSUF=obj where the other two set o. It is stated once here. The CUDA backend
# remains Linux-only.
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
  EXESUF        = .exe
  IMPLIBSUF     = lib
  CXXSTD        = -std=c++14
  # The UCRT marks getenv & co. "unsafe" and warns on every include of
  # threadpool.inl; the portable spelling is the point here.
  PLATFORMFLAGS = -D_CRT_SECURE_NO_WARNINGS
  PICFLAG       =
  SONAME_PREFIX =
  RPATH         =
  NO_UNDEFINED  =
endif

# How a consumer links one of our shared libraries, $(call LINKLIB,<dir>,<name>)
# with <name> e.g. fastfields-cpu. `-L -l` everywhere but Windows, where the
# MSVC-target driver would turn `-lfastfields-cpu` into `fastfields-cpu.lib`
# (no `lib` prefix) -- so the import library is named by path instead.
ifdef IS_WINDOWS
  LINKLIB = $(1)/lib$(2).$(IMPLIBSUF)
else
  LINKLIB = -L$(1) -l$(2)
endif

########################################################################
#	Default goal
########################################################################

# Every Makefile that includes this fragment defines an `all` target, and each
# includes it *before* declaring any target of its own. Without the explicit
# assignment below, GNU make would take the first target it sees -- the output
# directory rule immediately after this block -- as the default goal, so a bare
# `make` would silently create build/ and exit 0 having built nothing.
#
# That is not hypothetical: it is what the consolidated tree did on arrival,
# and it broke fastfields-dlpack, whose setup.py shells out to
# `make -C _fastfields_lib CXX=...` with no explicit target and then fails on
# the missing .so files. The pre-consolidation hub Makefile happened to declare
# `all: lib` as its first target, so it did not need this.
.DEFAULT_GOAL := all

########################################################################
#	Output directories
########################################################################

$(BUILDDIR) $(LIBDIR) $(OBJDIR) $(TESTDIR):
	$(MKDIR) $@
