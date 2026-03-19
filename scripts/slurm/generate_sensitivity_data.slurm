#!/bin/bash
#SBATCH --job-name=gen_sensitivity
#SBATCH --account=NN12048K
#SBATCH --time=02:00:00
#SBATCH --partition=accel
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/gen_sensitivity_%j.out
#SBATCH --error=logs/gen_sensitivity_%j.err

set -o errexit
set -o nounset

echo "=== Generating sensitivity datasets ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

module purge
module load NRIS/GPU

if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

CONTAINER="/cluster/work/support/container/pytorch_nvidia_24.12_extended.sif"
OUTPUT_DIR="/cluster/home/michechi/MIMICIV/data/simulation/sensitivity/"

srun apptainer exec \
    --bind /cluster/work/projects/nn12048k:/cluster/work/projects/nn12048k \
    --bind /cluster/home/michechi:/cluster/home/michechi \
    "$CONTAINER" \
    python3 -u -m simulation.generate_sensitivity_datasets \
        --all \
        --output_dir "$OUTPUT_DIR"

echo "=== DONE ==="
