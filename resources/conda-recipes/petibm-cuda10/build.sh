#! /bin/sh
#
# build_amgx.sh
# Copyright (C) 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.
#

set -e

ln -sf ${PREFIX}/lib ${PREFIX}/lib64

mkdir build
cd build

cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS_USED}" \
    -DCMAKE_CUDA_HOST_COMPILER=${CXX} \
    -DCMAKE_CUDA_ARCHITECTURES="all" \
    -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS_USED}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
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
rm ${PREFIX}/lib64
