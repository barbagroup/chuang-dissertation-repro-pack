#! /bin/sh
#
# job.sh
# Copyright (C) 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.
#

# get the path to this script (method depending on whether using Slurm)
SCRIPTPATH=$(realpath $0)

# get the path to the directory based on where this script is in
export ROOT=$(dirname ${SCRIPTPATH})

# get current time (seconds since epich)
export TIME=$(date +"%s")

# create a log file
mkdir -p ${ROOT}/logs
export LOG=${ROOT}/logs/run-${TIME}.log
echo "Current epoch time: ${TIME}" >> ${LOG}
echo "Case folder: ${ROOT}" >> ${LOG}
echo "Job script: ${SCRIPTPATH}" >> ${LOG}

echo "" >> ${LOG}
echo "===============================================================" >> ${LOG}
lscpu 2>&1 >> ${LOG}
echo "===============================================================" >> ${LOG}

echo "" >> ${LOG}
echo "===============================================================" >> ${LOG}
nvidia-smi -L 2>&1 >> ${LOG}
echo "===============================================================" >> ${LOG}

# run with gpus
echo "Start the run" >> ${LOG}

python ${ROOT}/main.py 2>&1 > ${LOG}
