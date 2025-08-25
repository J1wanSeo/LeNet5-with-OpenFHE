cd build
rm -rf *
cmake .. -DOpenFHE_WITH_INTEL_HEXL=ON -DOpenFHE_INTEL_HEXL_HINT_DIR=/usr/local/lib
make clean
make -j
if [ -n "$1" ]; then
    ./conv_bn_exec "$1"
else
    ./conv_bn_exec
fi
