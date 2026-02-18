#!/bin/bash

#SBATCH --job-name=XGB_fraction
#SBATCH --account=ec12
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/root/MIMICIV/output_xgb_fraction_%j.out

source ${HOME}/.bashrc

# Exit on error and disallow unset variables
set -o errexit
set -o nounset

# Load required modules
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8

# Run the experiment with both encodings
python3 /root/MIMICIV/XGBoost_fraction_experiment.py \
    --csv_to_use 9 \
    --path_csv /root/MIMICIV/data/simulation/ \
    --output_dir /root/MIMICIV/results/ \
    --model_name meta-llama/Llama-3.1-8B \
    --seed 9550 \
    --run_basic \
    --run_llm \
    ${@}
