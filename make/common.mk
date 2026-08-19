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
SONAME     = soname
OMPFLAG    = -fopenmp
USE_OPENMP ?= 0

# Position-independent code + shared-library soname flags. Both are POSIX-only:
# on Windows the PE/COFF linker has no soname and code is position-independent
# by default, so they are cleared in the Windows block below and referenced via
# these variables (never hard-coded) in the link rules.
PICFLAG       = -fPIC
SONAME_PREFIX = -Wl,-$(SONAME),
SONAME_FLAG   = $(if $(SONAME_PREFIX),$(SONAME_PREFIX)$(@F))
RPATH         = -Wl,-rpath,'$$ORIGIN'/../lib

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
ifeq (Darwin,$(PLATFORM))
  OMPFLAG    = -fopenmp=libiomp5
  SOSUF      = dylib
  SONAME     = install_name
  RPATH      = -Wl,-rpath,@loader_path/../lib
endif
ifeq (arm64,$(PLATFORM))
  OMPFLAG    = -fopenmp=libiomp5
  SOSUF      = dylib
  SONAME     = install_name
  RPATH      = -Wl,-rpath,@loader_path/../lib
endif

##### Windows (native OS=Windows_NT, or a Unix-like shell: MINGW*/MSYS) #####
# Built with clang++ (as CI does): objects stay .o, shared libs are .dll, and
# there is no soname / -fPIC / rpath (the PE/COFF linker rejects -Wl,-soname and
# code is position-independent by default). A Unix-like shell (git bash / MSYS)
# is required so the recipe commands (cp/rm/mkdir/uname) resolve.
#
# This block used to be triplicated across the three repos and had already
# drifted -- cuda-lib's copy knew nothing about IS_WINDOWS or PICFLAG and set
# MOSUF=obj where the other two set o. It is stated once here, in cpu-lib's
# (non-stale) form. Nothing implements Windows support; this only keeps the
# door open.
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
  RPATH         =
endif

########################################################################
#	Output directories
########################################################################

$(BUILDDIR) $(LIBDIR) $(OBJDIR) $(TESTDIR):
	$(MKDIR) $@
