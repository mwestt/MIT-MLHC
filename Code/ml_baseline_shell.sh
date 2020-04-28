#!/bin/bash
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 60 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=32G # Memory in GB (see also --mem-per-cpu)
#SBATCH -o output_%j.out # Standard out goes to this file
#SBATCH -e error_%j.err # Standard err goes to this file
# LOAD_MODULES
module load Anaconda3/2019.10

python ml_baseline.py

echo JOB_FINISHED