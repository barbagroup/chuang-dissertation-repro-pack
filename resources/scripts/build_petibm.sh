#! /bin/sh
#
# build_amgx.sh
# Copyright (C) 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.
#

set -e

if [[ -z ${CONDA_PREFIX} ]]; then
    echo "Requiring being in a Conda environment."
    exit 1;
else
    echo "Using conda env at ${CONDA_PREFIX}"
    echo "NVCC: $(which nvcc)"
fi

# get the path to the directory based on where this script is in
export CURDIR=$(pwd)
export ROOT=$(dirname $(realpath $0))/tmps
export SRCDIR=${HOME}/Sync/repos/PetIBM
export BUILDDIR=${ROOT}/builds
export INSTALLDIR=${ROOT}/installs
rm -rf ${BUILDDIR}/petibm && mkdir -p ${BUILDDIR}/petibm

cd ${BUILDDIR}/petibm

cmake \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_CUDA_HOST_COMPILER=${CC} \
    -DCMAKE_CUDA_ARCHITECTURES="35" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/petibm \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -DPETSC_DIR=${INSTALLDIR}/petsc \
    -DPETSC_ARCH="" \
    -DCUDA_DIR=${CONDA_PREFIX} \
    -DAMGX_DIR=${INSTALLDIR}/amgx \
    -DPETIBM_ENABLE_TESTS=ON \
    -DPETIBM_USE_AMGX=ON \
    -DPETIBM_BUILD_YAMLCPP=ON \
    -DPETIBM_BUILD_AMGXWRAPPER=ON \
    ${SRCDIR}

    #-DAMGXWRAPPER_DIR=${INSTALLDIR}/amgxwrapper \
    #-DYAMLCPP_DIR=${INSTALLDIR}/yaml-cpp \
make all -j $(nproc)
make check
make install

cd ${CURDIR}
