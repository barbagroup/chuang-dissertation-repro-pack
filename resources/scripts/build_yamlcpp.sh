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
export COMMIT="420c98231094b1cd2e5de3a714c4e3ee9b4f1118"
export CURDIR=$(pwd)
export ROOT=$(dirname $(realpath $0))/tmps
export SRCDIR=${ROOT}/sources
export BUILDDIR=${ROOT}/builds
export INSTALLDIR=${ROOT}/installs
mkdir -p ${SRCDIR}/yaml-cpp
mkdir -p ${BUILDDIR}/yaml-cpp

curl \
    -L https://github.com/jbeder/yaml-cpp/tarball/${COMMIT} \
    -o ${SRCDIR}/yaml-cpp.tar.gz

tar -xf ${SRCDIR}/yaml-cpp.tar.gz -C ${SRCDIR}/yaml-cpp --strip-component=1

cd ${BUILDDIR}/yaml-cpp

cmake \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_CXX_FLAGS="--std=c++14" \
    -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/yaml-cpp \
    -DCMAKE_VERBOSE_MAKEFILE=OFF  \
    -DCMAKE_COLOR_MAKEFILE=ON  \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_BUILD_SHARED_LIBS=ON \
    ${SRCDIR}/yaml-cpp

make -j 12
make install

cd ${CURDIR}
