define verb
	@ echo "_____________________________________________________________"
	@ echo ""
	@ echo "        " $(1)
	@ echo "_____________________________________________________________"
	@ echo ""
endef

########################################################################
# 	Options
########################################################################

COPY        ?= cp -f
DEL         ?= rm -f
MOVE        ?= mv -f
MKDIR     	?= mkdir -p
BUILDDIR  	?= ./build
CXXFLAGS  	+= -std=c++17 -O3 -ferror-limit=1 -ftemplate-backtrace-limit=0
INCLUDES  	+=
TESTFLAGS 	+= -ferror-limit=1 -ftemplate-backtrace-limit=0
UNAME     	?= uname
GET_ARCH  	?= $(UNAME) -m
MOSUF 	  	 = o
SOSUF      	 = so
SONAME     	 = soname
OMPFLAG    	 = -fopenmp
RPATH        = -Wl,-rpath,'$$ORIGIN'/../lib
USE_OPENMP 	?= 0
# Build the CUDA backend and link it in (needs nvcc + the cuda submodule).
# Default off: the CPU path is the tested source of truth and CI has no GPU.
USE_CUDA   	?= 0
# Position-independent code + shared-library soname flags. Both are POSIX-only:
# on Windows the PE/COFF linker has no soname and code is position-independent
# by default, so they are cleared in the Windows block below and referenced via
# these variables (never hard-coded) in the link rules.
PICFLAG       = -fPIC
SONAME_PREFIX = -Wl,-$(SONAME),
# Full soname linker flag for the shared lib, empty on Windows (see below):
# e.g. -Wl,-soname,libfastfields.so. $(@F) is the target's file name,
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
# 	CUDA backend (optional)
########################################################################

# When USE_CUDA=1, compile the lib against the CUDA declarations and link the
# cuda backend library. pushpull is excluded from the cuda-lib MODULES for now
# (slow compile), so its CUDA path is compiled out here too via
# FF_CUDA_NO_PUSHPULL to keep the link resolved (see MIGRATION.md T21).
ifeq ($(USE_CUDA),1)
  CXXFLAGS    += -DFF_WITH_CUDA -DFF_CUDA_NO_PUSHPULL
  CUDA_LDFLAGS = -L$(BUILDDIR)/lib -lfastfields-cuda
  CUDA_DEP     = $(BUILDDIR)/lib/libfastfields-cuda.$(SOSUF)
endif

########################################################################
# 	Public Targets
########################################################################

all: lib

clean: clean-lib clean-obj clean-cpu

.PHONY: all clean

########################################################################
# 	Build directory
########################################################################

$(BUILDDIR):
	$(MKDIR) $(BUILDDIR)

########################################################################
# 	Clean
########################################################################

clean-obj:
	$(DEL) $(BUILDDIR)/*.$(MOSUF)

clean-lib:
	$(DEL) $(BUILDDIR)/*.$(SOSUF)

clean-cpu:
	$(MAKE) -C cpu clean

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

lib: \
	verb.build.lib \
  libcpu \
  $(CUDA_DEP) \
	$(BUILDDIR)/libfastfields.$(SOSUF) \
	verb.build.lib.done

libcpu: $(BUILDDIR)/lib/libfastfields-cpu.$(SOSUF)

# Absolute path: the cpu/cuda submodules are symlinks, and `make -C` resolves
# them to their physical dir, so a relative PREFIX (../build) would install into
# the wrong tree. $(abspath) pins it to this repo's build/.
export PREFIX = $(abspath $(BUILDDIR))

$(BUILDDIR)/lib/libfastfields-cpu.$(SOSUF): $(BUILDDIR)
	$(MAKE) -C cpu install

# Build + install the CUDA backend (nvcc) into build/lib. Skipped unless
# USE_CUDA=1 pulls it in via $(CUDA_DEP).
$(BUILDDIR)/lib/libfastfields-cuda.$(SOSUF): $(BUILDDIR)
	$(MAKE) -C cuda install

$(BUILDDIR)/libfastfields.$(SOSUF): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared $(PICFLAG) $(SONAME_FLAG) $(RPATH) \
  -L$(BUILDDIR)/lib -lfastfields-cpu $(CUDA_LDFLAGS) \
  -o $@ $^

########################################################################
# 	Objects
########################################################################

$(BUILDDIR)/%.$(MOSUF): %.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
  $(PICFLAG) -c -o $@ $<

########################################################################
# 	Messages
########################################################################

verb.build.lib:
	$(call verb, "Building library...")

verb.build.lib.done:
	$(call verb, "Building library: done.")
