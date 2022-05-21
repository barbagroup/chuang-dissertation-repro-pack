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
export COMMIT="release-1.11.0"
export CURDIR=$(pwd)
export ROOT=$(dirname $(realpath $0))/tmps
export SRCDIR=${ROOT}/sources
export BUILDDIR=${ROOT}/builds
export INSTALLDIR=${ROOT}/installs
mkdir -p ${SRCDIR}/gtest
mkdir -p ${BUILDDIR}/gtest

curl \
    -L https://github.com/google/googletest/tarball/${COMMIT} \
    -o ${SRCDIR}/gtest.tar.gz

tar -xf ${SRCDIR}/gtest.tar.gz -C ${SRCDIR}/gtest --strip-component=1

cd ${BUILDDIR}/gtest

cmake \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_CXX_FLAGS="--std=c++14" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/gtest \
    -DCMAKE_VERBOSE_MAKEFILE=OFF  \
    -DCMAKE_COLOR_MAKEFILE=OFF  \
    -DBUILD_GMOCK=OFF \
    -DINSTALL_GTEST=ON \
    ${SRCDIR}/gtest

make -j 12
make install

cd ${CURDIR}
