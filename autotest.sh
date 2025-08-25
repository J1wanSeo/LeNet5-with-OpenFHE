cd build
rm -rf *
cmake .. -DWITH_INTEL_HEXL=ON -DINTEL_HEXL_HINT_DIR=/usr/local/lib
make clean
make -j
if [ -n "$1" ]; then
    ./conv_bn_exec "$1"
else
    ./conv_bn_exec
fi
