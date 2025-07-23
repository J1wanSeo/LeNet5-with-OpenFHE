cd build
rm -rf *
cmake ..
make clean
make -j
./conv_bn_exec
#./fc_test
