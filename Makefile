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
RPATH        = -Wl,-rpath,$$ORIGIN/../lib
USE_OPENMP 	?= 0

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
	distance

OBJECTS  = $(addprefix $(BUILDDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
CPPFILES = $(addsuffix .cpp,$(MODULES))

lib: \
	verb.build.lib \
  libcpu \
	$(BUILDDIR)/libfastfields.$(SOSUF) \
	verb.build.lib.done

libcpu: $(BUILDDIR)/lib/libfastfields-cpu.$(SOSUF)

export PREFIX = ../$(BUILDDIR)

$(BUILDDIR)/lib/libfastfields-cpu.$(SOSUF): $(BUILDDIR)
	$(MAKE) -C cpu install

$(BUILDDIR)/libfastfields.$(SOSUF): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -Wl,-$(SONAME),libfastfields.$(SOSUF) $(RPATH) \
  -L$(BUILDDIR)/lib -lfastfields-cpu \
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
