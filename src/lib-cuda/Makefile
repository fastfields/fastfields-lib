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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Boundary-condition compile policy  (see kernels/bounds.h)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Every boundary condition used to be a compile-time template parameter, so
# each regulariser kernel was instantiated 8 times per (ndim, dtype, offset_t)
# combination. On the CUDA side that pushes `ptxas` past 16 GB of RAM -- it gets
# OOM-killed on a standard 16 GB CI runner.
#
# `bound::type::Dynamic` provides the same operator with the condition read at
# run time: one instantiation shared by every condition that does not keep a
# static fast path. The default below keeps dedicated instantiations for two
# conditions and routes the other six through the Dynamic implementation:
#   DCT2 -- Neumann, the library-wide default boundary condition, and the
#           symmetric case (no sign flip at the boundary);
#   DST2 -- Dirichlet, the antisymmetric case, so the sign-flipping code path
#           (`_sign::periodic2` + the sign-aware `cget`/`add` selected by
#           FF_ISO_SIGN) also keeps a real compile-time instantiation rather
#           than only ever being reached through the shared Dynamic one.
#
# Whoever compiles the library can trade compile cost against per-voxel speed:
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=1"                     # all 8 static
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0"                     # all dynamic
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DFT=1 \
#                    -DFF_STATIC_BOUND_DCT2=1"                 # pick your own
# Results are identical either way; only code size, compile cost and speed move.
# Kept out of CXXFLAGS on purpose so that overriding CXXFLAGS on the command
# line (as CI does, to force -O1) does not silently drop the policy.
BOUNDFLAGS  ?= -DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DCT2=1 -DFF_STATIC_BOUND_DST2=1

# Interpolation-order compile policy (see kernels/spline.h) -- the same idea one
# axis further out. `pushpull` templates on the spline order as well as the
# boundary condition (and ndim, dtype, offset_t), so all eight orders static
# would multiply the already-Dynamic-bound matrix by up to eight again.
#
# `spline::type::Dynamic` provides the same operator with the order read at run
# time. The default below keeps dedicated instantiations for the four lowest
# (and by far the most commonly used) orders -- Nearest/Linear/Quadratic/Cubic
# -- and routes FourthOrder-SeventhOrder through the Dynamic implementation:
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=1"                    # all 8 static
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=0"                    # all dynamic
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=0 \
#                     -DFF_STATIC_SPLINE_CUBIC=1"               # pick your own
# Results are identical either way; only code size, compile cost and speed move.
# Kept out of CXXFLAGS for the same reason as BOUNDFLAGS.
SPLINEFLAGS ?= -DFF_STATIC_SPLINES=0 \
               -DFF_STATIC_SPLINE_NEAREST=1 -DFF_STATIC_SPLINE_LINEAR=1 \
               -DFF_STATIC_SPLINE_QUADRATIC=1 -DFF_STATIC_SPLINE_CUBIC=1

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

# The regularisers are split into a core module and an `_rls` module (the
# reweighted-least-squares ops) -- a CUDA-only split that cpu-lib does not need.
# Rationale: measured with the default BOUNDFLAGS, `ptxas` peaks at ~3.8 GB per
# split module but ~6-7 GB for the combined file. Two ~4 GB jobs fit a 16 GB CI
# runner under `make -j2`; two ~7 GB ones do not. The split is a memory/parallelism
# measure, not a correctness one -- the boundary-condition policy above is what
# actually brought this build back from a ~16 GB ptxas OOM.
MODULES = \
	distance \
	reg_field \
	reg_field_rls \
	reg_flow \
	reg_flow_rls \
	pushpull \
	pushpull_backward

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
	$(NVCC) $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) $(INCLUDES) -x cu -Xcompiler -fPIC -c -o $@ $<

########################################################################
# 	Messages
########################################################################

verb.build.lib:
	$(call verb, "Building CPU library...")

verb.build.lib.done:
	$(call verb, "Building CPU library: done.")
