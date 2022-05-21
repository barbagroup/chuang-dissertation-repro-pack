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
fi

# get the path to the directory based on where this script is in
export VERSION="3.16.6"
export CURDIR=$(pwd)
export ROOT=$(dirname $(realpath $0))/tmps
export SRCDIR=${ROOT}/sources
export INSTALLDIR=${ROOT}/installs
mkdir -p ${SRCDIR}/petsc

curl \
    -L http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${VERSION}.tar.gz \
    -o ${SRCDIR}/petsc.tar.gz

tar -xf ${SRCDIR}/petsc.tar.gz -C ${SRCDIR}/petsc --strip-component=1

# petsc's configure must run under the source directory
cd ${SRCDIR}/petsc

./configure \
    CFLAGS="${CFLAGS}" \
    CXXFLAGS="${CXXFLAGS}" \
    CPP="${CPP}" \
    CPPFLAGS="${CPPFLAGS}" \
    LDFLAGS="${LDFLAGS}" \
    AR="${AR}" \
    RANLIB="${RANLIB}" \
    --prefix="${INSTALLDIR}/petsc" \
    --with-petsc-arch="for-petibm" \
    --with-default-arch=0 \
    --with-fortran-bindings=0 \
    --with-precision=double \
    --with-clanguage=C \
    --with-shared-libraries=1 \
    --with-cc=mpicc \
    --with-cxx=mpicxx \
    --with-fc=0 \
    --with-pic=1 \
    --with-cxx-dialect=C++14 \
    --with-debugging=1 \
    --with-gcov=0 \
    --with-mpi=1 \
    --with-hypre=1 \
    --with-hypre-dir=${CONDA_PREFIX} \
    --with-hdf5=1 \
    --with-hdf5-dir=${CONDA_PREFIX} \
    --with-superlu_dist=1 \
    --with-superlu_dist-dir=${CONDA_PREFIX} \
    --with-blaslapack-dir=${CONDA_PREFIX}

make PETSC_DIR=${SRCDIR}/petsc PETSC_ARCH=for-petibm all
make PETSC_DIR=${SRCDIR}/petsc PETSC_ARCH=for-petibm install
make PETSC_DIR=${INSTALLDIR}/petsc PETSC_ARCH="" check

cd ${CURDIR}
