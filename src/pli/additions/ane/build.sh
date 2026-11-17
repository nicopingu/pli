#!/bin/bash
# Build native/libpli_ane.dylib on the machine (clang++ of the Command Line Tools; no Metal compiler needed: the Neural Engine
# program is compiled by the service, the GPU kernels are compiled on the fly by MLX).
set -e; cd "$(dirname "$0")/native"
xcrun -sdk macosx clang++ -std=c++20 -O2 -Wall -Wextra -fobjc-arc -dynamiclib -I. pli_ane.mm ane/Program.mm \
  -framework Foundation -framework Metal -framework IOSurface -o libpli_ane.dylib
echo "libpli_ane.dylib: $(nm -gU libpli_ane.dylib | grep -c pli_ane_) functions"
# PLI (21/10/2026): requantization of the weights of the Neural Engine part on the CPU (requant_cpu.py, PLI_ANE_REQUANT_CPU)
xcrun -sdk macosx clang -std=c17 -O3 -Wall -Wextra -ffp-contract=off -dynamiclib requant_cpu.c -o libpli_rq.dylib
echo "libpli_rq.dylib: $(nm -gU libpli_rq.dylib | grep -c pli_rq_) functions"
