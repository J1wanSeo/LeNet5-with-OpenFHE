cd build
rm -rf *
cmake ..
make clean
make -j
if [ -n "$1" ]; then
    ./conv_bn_exec "$1"
else
    ./conv_bn_exec
fi
