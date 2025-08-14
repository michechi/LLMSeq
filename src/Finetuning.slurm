#!/bin/bash

#SBATCH --job-name=project_m
#SBATCH --account=ec12
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=accel --gpus=1
#SBATCH --exclude=gpu-14
#SBATCH --exclude=gpu-4
#SBATCH --nodelist=gpu-8,gpu-9
# Increase this number when you really need parallel computing
#SBATCH --cpus-per-task=16

#source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge 
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8 
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8 
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8 
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8 
module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8
module load nlpl-bitsandbytes/0.42.0-foss-2022b-Python-3.10.8

# Setup monitoring
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
	--format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process

# We pass as arguments the path to the training corpus and (optionally) the number of CPU cores we want to use
python3 Finetuning_rfct.py ${@}

# After computation stop monitoring
kill -SIGINT "$NVIDIA_MONITOR_PID"