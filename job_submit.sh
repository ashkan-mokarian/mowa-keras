#!/bin/bash
env
#$ -l gpu=1 -l h_gpu=1
#$ -l m_mem_free=90G
#$ -cwd
#$ -V
#$ -e error_log_$JOB_ID
#$ -o out_log_$JOB_ID
#$ -l h_rt=96:00:00

./run_experiment.sh "$@"
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
        exit 100
fi
