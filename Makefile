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
CXXFLAGS  	+= -std=c++11 -O3 -ferror-limit=1 -ftemplate-backtrace-limit=0
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

########################################################################
# 	Platform-specific settings
########################################################################

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

##### Windows #####
ifeq (MINGW32,$(word 1,$(subst _, ,$(PLATFORM)))) # MSVC
  MOSUF      = obj
  SOSUF      = dll
  OMPFLAG    = /openmp
  RPATH      =
endif
ifeq (MINGW64,$(word 1,$(subst _, ,$(PLATFORM)))) # MSVC
  MOSUF      = obj
  SOSUF      = dll
  RPATH      =
endif
ifeq (MSYS,$(word 1,$(subst _, ,$(PLATFORM)))) # GCC
  MOSUF      = obj
  SOSUF      = dll
  RPATH      =
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

export PREFIX = ../$(BUILDDIR)

$(BUILDDIR)/lib/libfastfields-cpu.$(SOSUF): $(BUILDDIR)
	$(MAKE) -C cpu install

# Build + install the CUDA backend (nvcc) into build/lib. Skipped unless
# USE_CUDA=1 pulls it in via $(CUDA_DEP).
$(BUILDDIR)/lib/libfastfields-cuda.$(SOSUF): $(BUILDDIR)
	$(MAKE) -C cuda install

$(BUILDDIR)/libfastfields.$(SOSUF): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -Wl,-$(SONAME),libfastfields.$(SOSUF) $(RPATH) \
  -L$(BUILDDIR)/lib -lfastfields-cpu $(CUDA_LDFLAGS) \
  -o $@ $^

########################################################################
# 	Objects
########################################################################

$(BUILDDIR)/%.$(MOSUF): %.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
  -fPIC -c -o $@ $<

########################################################################
# 	Messages
########################################################################

verb.build.lib:
	$(call verb, "Building library...")

verb.build.lib.done:
	$(call verb, "Building library: done.")
