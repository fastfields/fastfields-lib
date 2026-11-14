#pragma once
// Symbol visibility for the two shared libraries' public entry points.
//
//   FF_CPU_API  marks a function libfastfields-cpu defines (api/cpu/*.h)
//   FF_API      marks a function the hub, libfastfields, defines (api/*.h)
//
// Each expands to "exported" in the translation units that DEFINE the
// function and to "imported" everywhere else. Which one a translation unit is
// is said by the build, never by the source:
//
//   -DFF_BUILDING_CPU_LIB   src/lib-cpu's objects  (src/lib-cpu/Makefile)
//   -DFF_BUILDING_LIB       src/lib's objects      (src/lib/Makefile)
//   -DFF_STATIC_LINK        no shared library at all: the test executables,
//                           which link the module objects in directly
//
// Why this exists: a Windows DLL exports nothing unless told to, so before
// these macros libfastfields-cpu.dll had an empty export table, no import
// library was written for it, and nothing -- the hub, the Python extension --
// could link against it (fastfields-dlpack#8). `__declspec(dllimport)` on the
// consumer side is what lets the compiler call through the import table
// directly; MSVC's `cl` and clang (MSVC or MinGW target) both spell it so.
//
// Everywhere else the macros spell default visibility, which is already what
// every symbol has (nothing here builds with -fvisibility=hidden), so on
// Linux and macOS they change nothing today. They are not empty there on
// purpose: if a library is ever built with hidden visibility, these are
// exactly the symbols that must stay visible.
//
// The CUDA backend (api/cuda/*.h) is Linux-only and is not annotated.

#if defined(_WIN32) || defined(__CYGWIN__)
#  define FF_DLL_EXPORT __declspec(dllexport)
#  define FF_DLL_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#  define FF_DLL_EXPORT __attribute__((visibility("default")))
#  define FF_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define FF_DLL_EXPORT
#  define FF_DLL_IMPORT
#endif

#if defined(FF_STATIC_LINK)
#  define FF_CPU_API
#  define FF_API
#else
#  if defined(FF_BUILDING_CPU_LIB)
#    define FF_CPU_API FF_DLL_EXPORT
#  else
#    define FF_CPU_API FF_DLL_IMPORT
#  endif
#  if defined(FF_BUILDING_LIB)
#    define FF_API FF_DLL_EXPORT
#  else
#    define FF_API FF_DLL_IMPORT
#  endif
#endif
