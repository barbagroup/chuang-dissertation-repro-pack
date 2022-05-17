#! /bin/sh
#
# build_amgx.sh
# Copyright (C) 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.
#

set -e

ln -sf $PREFIX/lib/stubs/libcuda.so $PREFIX/lib/libcuda.so
ln -sf $PREFIX/lib $PREFIX/lib64

mkdir build
cd build

cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_CXX_FLAGS="-std=c++14" \
    -DCMAKE_CUDA_HOST_COMPILER=${CXX} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -DCMAKE_COLOR_MAKEFILE=OFF \
    -DCUDA_NVCC_FLAGS_RELEASE="" \
    -DCUDA_ARCH="35 70" \
    -DCMAKE_NO_MPI=OFF \
    -DAMGX_NO_RPATH=OFF \
    ../

make -j ${CPU_COUNT}
make install

cd ..
rm $PREFIX/lib64
rm $PREFIX/lib/libcuda.so
