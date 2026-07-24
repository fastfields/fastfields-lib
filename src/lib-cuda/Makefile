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
# nvcc flags: -std/-O3 pass straight through. The clang-only diagnostic flags
# (-ferror-limit / -ftemplate-backtrace-limit) are NOT understood by nvcc, and
# nvcc 12 builds these CUDA headers under -std=c++14.
CXXFLAGS  	+= -std=c++14 -O3
INCLUDES  	+=
TESTFLAGS 	+= -ferror-limit=1 -ftemplate-backtrace-limit=0
UNAME     	?= uname
GET_ARCH  	?= $(UNAME) -m
MOSUF 	  	 = o
SOSUF      	 = so
SONAME     	 = soname
# nvcc from PATH (apt's nvidia-cuda-toolkit installs /usr/bin/nvcc). Override
# NVCC, or put a CUDA toolkit's bin on PATH, for a non-standard install. The
# previous $(dir $(dir ...)) derivation produced /usr/bin//bin/nvcc for an
# apt install (the trailing-slash makes the second $(dir) a no-op).
NVCC        ?= nvcc

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
  SOSUF      = dylib
  SONAME     = install_name
endif
ifeq (arm64,$(PLATFORM))
  SOSUF      = dylib
  SONAME     = install_name
endif

##### Windows #####
ifeq (MINGW32,$(word 1,$(subst _, ,$(PLATFORM)))) # MSVC
  MOSUF      = obj
  SOSUF      = dll
  OMPFLAG    = /openmp
endif
ifeq (MINGW64,$(word 1,$(subst _, ,$(PLATFORM)))) # MSVC
  MOSUF      = obj
  SOSUF      = dll
endif
ifeq (MSYS,$(word 1,$(subst _, ,$(PLATFORM)))) # GCC
  MOSUF      = obj
  SOSUF      = dll
endif

########################################################################
# 	Public Targets
########################################################################

all: libcuda

install: libcuda | $(PREFIX)/lib
	$(COPY) $(BUILDDIR)/libfastfields-cuda.$(SOSUF) $(PREFIX)/lib

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
	distance

OBJECTS  = $(addprefix $(BUILDDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
CPPFILES = $(addsuffix .cpp,$(MODULES))

libcuda: \
	verb.build.lib \
	$(BUILDDIR)/libfastfields-cuda.$(SOSUF) \
	verb.build.lib.done

$(BUILDDIR)/libfastfields-cuda.$(SOSUF): $(OBJECTS)
	$(NVCC) $(CXXFLAGS) -shared -Xcompiler -fPIC -Xlinker -$(SONAME)=libfastfields-cuda.$(SOSUF) -o $@ $^

########################################################################
# 	Objects
########################################################################

$(BUILDDIR)/%.$(MOSUF): %.cpp | $(BUILDDIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -x cu -Xcompiler -fPIC -c -o $@ $<

########################################################################
# 	Messages
########################################################################

verb.build.lib:
	$(call verb, "Building CPU library...")

verb.build.lib.done:
	$(call verb, "Building CPU library: done.")
