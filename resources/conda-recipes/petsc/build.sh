#! /bin/bash
#
# build_amgx.sh
# Copyright (C) 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.
#

set -e

# get an updated config.sub and config.guess
cp $BUILD_PREFIX/share/gnuconfig/config.* .

export PETSC_DIR=${SRC_DIR}
export PETSC_ARCH="for-petibm"

export OMPI_CC=${CC}
export OMPI_CXX=${CXX}
export OMPI_MCA_plm=isolated
export OMPI_MCA_rmaps_base_oversubscribe=yes
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OPAL_PREFIX=$PREFIX

export CC=mpicc
export CXX=mpicxx

# scrub debug-prefix-map args, which cause problems in pkg-config
export CFLAGS=$(echo ${CFLAGS:-} | sed -E 's@\-fdebug\-prefix\-map[^ ]*@@g')
export CXXFLAGS=$(echo ${CXXFLAGS:-} | sed -E 's@\-fdebug\-prefix\-map[^ ]*@@g')

python ./configure \
    CFLAGS="${CFLAGS}" \
    CXXFLAGS="${CXXFLAGS}" \
    CPP="${CPP}" \
    CPPFLAGS="${CPPFLAGS}" \
    LDFLAGS="${LDFLAGS}" \
    AR="${AR}" \
    RANLIB="${RANLIB}" \
    --prefix="${PREFIX}" \
    --with-petsc-arch=${PETSC_ARCH} \
    --with-default-arch=0 \
    --with-precision=double \
    --with-clanguage=C \
    --with-shared-libraries=1 \
    --with-pic=1 \
    --with-debugging=0 \
    --with-gcov=0 \
    --with-cc=mpicc \
    --with-clib-autodetect=0 \
    --COPTFLAGS="-O3" \
    --with-cxx=mpicxx \
    --with-cxxlib-autodetect=0 \
    --with-cxx-dialect=C++14 \
    --CXXOPTFLAGS="-O3" \
    --with-fc=0 \
    --with-fortran-bindings=0 \
    --with-fortranlib-autodetect=0 \
    --with-mpi=1 \
    --with-hypre=1 \
    --with-hypre-dir=${PREFIX} \
    --with-hdf5=1 \
    --with-hdf5-dir=${PREFIX} \
    --with-superlu_dist=1 \
    --with-superlu_dist-dir=${PREFIX} \
    --with-blaslapack-dir=${PREFIX}

# verify that gcc_ext isn't linked
for f in $PETSC_ARCH/lib/petsc/conf/petscvariables $PETSC_ARCH/lib/pkgconfig/PETSc.pc; do
    if grep gcc_ext $f; then
        echo "gcc_ext found in $f"
        exit 1
    fi
done

# remove abspath of ${BUILD_PREFIX}/bin/python
sed -i "s%${PREFIX}/bin/python%python%g" $PETSC_ARCH/include/petscconf.h
sed -i "s%${PREFIX}/bin/python%python%g" $PETSC_ARCH/lib/petsc/conf/petscvariables
sed -i "s%${PREFIX}/bin/python%/usr/bin/env python%g" $PETSC_ARCH/lib/petsc/conf/reconfigure-${PETSC_ARCH}.py

# Replace abspath of ${PETSC_DIR} and ${BUILD_PREFIX} with ${PREFIX}
for path in $PETSC_DIR $BUILD_PREFIX; do
    for f in $(grep -l "${path}" $PETSC_ARCH/include/petsc*.h); do
        echo "Fixing ${path} in $f"
        sed -i "s%$path%\${PREFIX}%g" $f
    done
done

make PETSC_DIR=${SRC_DIR} PETSC_ARCH=for-petibm all
make PETSC_DIR=${SRC_DIR} PETSC_ARCH=for-petibm install

# Remove unneeded files
rm -f ${PREFIX}/lib/petsc/conf/configure-hash
find $PREFIX/lib/petsc -name '*.pyc' -delete

# replace ${BUILD_PREFIX} after installation,
# otherwise 'make install' above may fail
for f in $(grep -l "${BUILD_PREFIX}" -R "${PREFIX}/lib/petsc"); do
  echo "Fixing ${BUILD_PREFIX} in $f"
  sed -i "s%${BUILD_PREFIX}%${PREFIX}%g" $f
done

echo "Removing example files"
du -hs $PREFIX/share/petsc/examples/src
rm -fr $PREFIX/share/petsc/examples/src

echo "Removing data files"
du -hs $PREFIX/share/petsc/datafiles/*
rm -fr $PREFIX/share/petsc/datafiles
