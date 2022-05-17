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
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_CUDA_HOST_COMPILER=${CXX} \
    -DCMAKE_CUDA_ARCHITECTURES="all" \
    -DCMAKE_BUILD_TYPE=Release \
    -DYAMLCPP_DIR=${PREFIX} \
    -DPETSC_DIR=${PREFIX} \
    -DPETSC_ARCH="" \
    -DCUDA_DIR=${PREFIX} \
    -DAMGX_DIR=${PREFIX} \
    -DAMGXWRAPPER_DIR=${PREFIX} \
    -DPETIBM_ENABLE_TESTS=OFF \
    -DPETIBM_USE_AMGX=ON \
    ..

make all -j $(nproc)
make install

cd ..
