#! /bin/sh
#
# build_amgx.sh
# Copyright (C) 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.
#

set -e

mkdir build
cd build

cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_CUDA_HOST_COMPILER=${CXX} \
    -DCMAKE_CUDA_ARCHITECTURES="70;80" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -DCMAKE_COLOR_MAKEFILE=OFF \
    -DCUDA_ARCH="70;80" \
    -DCMAKE_NO_MPI=OFF \
    -DAMGX_NO_RPATH=OFF \
    ../

make -j ${CPU_COUNT}
make install

cd ..
