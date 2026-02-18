#!/bin/bash

#SBATCH --job-name=9llama1
#SBATCH --account=ec12
#SBATCH --time=24:00:00              # più tempo totale, perché tutti i run girano in serie
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/run_%A.out     # unico log con tutto dentro
#SBATCH --exclude=gpu-7

# ============================================================
# SANITY
# ============================================================
set -o errexit
set -o nounset

echo "=== Starting sequential fine-tuning job ==="
date

# ============================================================
# CACHE HANDLING (unica cache locale per tutto il job)
# ============================================================
export HF_HOME=$TMPDIR/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"
echo "Cache directory: $HF_HOME"

# ============================================================
# MODULES
# ============================================================
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8 
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8 
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8 
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8 
module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8
module load nlpl-bitsandbytes/0.42.0-foss-2022b-Python-3.10.8

# ============================================================
# CONFIG
# ============================================================
fractions=(0.01 0.10 0.30 0.50 0.75 1.0 ) #0.00001 0.0001 0.001 0.1 
tiny_type='1M'

# ============================================================
# MAIN LOOP
# ============================================================
for frac in "${fractions[@]}"; do
    echo "--------------------------------------------------------"
    echo "Starting training with fraction = ${frac}"
    echo "--------------------------------------------------------"
    date

    output_dir="src/output_slurms_fox/fraction_${frac}_tiny_${tiny_type}"
    mkdir -p "${output_dir}"

    python3 Finetuning_FOX_time_simulation_tiny.py \
        --fraction "${frac}" \
        --output_dir "${output_dir}" \
        ${@}

    echo "Completed fraction ${frac}"
    echo "========================================================"
    date
done

echo "All training runs completed!"
date