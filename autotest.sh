cd build
rm -rf *
cmake ..
make clean
make -j
#./test
./conv_bn_exec
#./fc_test
