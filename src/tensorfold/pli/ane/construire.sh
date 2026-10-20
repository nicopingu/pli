#!/bin/bash
# Compile natif/libpli_ane.dylib sur la machine (clang++ des Command Line Tools ; pas de compilateur Metal nécessaire : le programme
# du Neural Engine est compilé par le service, les noyaux GPU sont compilés à la volée par MLX).
set -e; cd "$(dirname "$0")/natif"
xcrun -sdk macosx clang++ -std=c++20 -O2 -Wall -Wextra -fobjc-arc -dynamiclib -I. pli_ane.mm ane/Program.mm \
  -framework Foundation -framework Metal -framework IOSurface -o libpli_ane.dylib
echo "libpli_ane.dylib : $(nm -gU libpli_ane.dylib | grep -c pli_ane_) fonctions"
