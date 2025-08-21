#!/bin/bash

#SBATCH --job-name=project_m
#SBATCH --account=ec12
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=accel --gpus=1
#SBATCH --exclude=gpu-14
#SBATCH --exclude=gpu-4
#SBATCH --nodelist=gpu-9
# Increase this number when you really need parallel computing
#SBATCH --cpus-per-task=16

#source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge 

source $HOME/ModernBERT/bin/activate

# # C) Una volta dentro il nodo:
# module purge
# echo "=== VERIFICA AMBIENTE ==="
# which python3
# python3 -V
# pip -V

# # D) Forza l'attivazione dell'ambiente con percorso assoluto
# source /fp/homes01/u01/ec-michelec/ModernBERT/bin/activate || echo "FAILED: Per favore verifica che $HOME punti a /fp/homes01/u01/ec-michelec"



# Setup monitoring
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
	--format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process

# We pass as arguments the path to the training corpus and (optionally) the number of CPU cores we want to use
python3 Finetuning_rfct.py ${@}

# After computation stop monitoring
kill -SIGINT "$NVIDIA_MONITOR_PID"