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

ln -sf ${CONDA_PREFIX}/lib ${CONDA_PREFIX}/lib64

# get the path to the directory based on where this script is in
export COMMIT="v1.6.1"
export CURDIR=$(pwd)
export ROOT=$(dirname $(realpath $0))/tmps
export SRCDIR=${ROOT}/sources
export BUILDDIR=${ROOT}/builds
export INSTALLDIR=${CONDA_PREFIX}
mkdir -p ${SRCDIR}/amgxwrapper
mkdir -p ${BUILDDIR}/amgxwrapper

curl \
    -L https://github.com/barbagroup/amgxwrapper/tarball/${COMMIT} \
    -o ${SRCDIR}/amgxwrapper.tar.gz

tar -xf ${SRCDIR}/amgxwrapper.tar.gz -C ${SRCDIR}/amgxwrapper --strip-component=1

cd ${BUILDDIR}/amgxwrapper

cmake \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_CUDA_HOST_COMPILER=${CC} \
    -DCMAKE_CUDA_ARCHITECTURES="all" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPETSC_DIR=${CONDA_PREFIX} \
    -DPETSC_ARCH="" \
    -DCUDA_DIR=${CONDA_PREFIX} \
    -DAMGX_DIR=${CONDA_PREFIX} \
    ${SRCDIR}/amgxwrapper

make all -j $(nproc)
make install

cd ${CURDIR}
