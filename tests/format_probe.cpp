// Scratch probe -- DO NOT MERGE. Exists only to prove two CI lints fire.
//
//   1. the quoted public include below must trip
//      `lint (source conventions)` -> normalise-include-delimiters (RED)
//   2. the deliberate mis-formatting must make
//      `lint (clang-format, changed lines)` report a diff and stay GREEN
#include "fastfields/core/defines.h"

namespace ff {
int    scratch_probe( int a,int b ) {
      if(a>b){return a  ;}
   return b;}
}  // namespace ff
