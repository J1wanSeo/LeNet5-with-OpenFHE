#!/bin/bash
set -e
cd build
rm -rf *
# cmake .. -DOpenFHE_WITH_INTEL_HEXL=ON -DOpenFHE_INTEL_HEXL_HINT_DIR=/usr/local/lib
cmake .. \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DWITH_OPENMP=ON
make clean
make -j
if [ -n "$1" ]; then
    ./conv_bn_exec "$1"
else
    ./conv_bn_exec
fi
