#!/bin/bash
#SBATCH --qos=short
#SBATCH --output=%x-%j.out
#SBATCH --job-name=patternscale
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-6  # one job per scenario in SCENARIOS (0-indexed)

# Submit from the patternscale repo root:  sbatch scripts/submit_downscaling.sh
# One-time setup in the environment:       pip install -e ".[io]"

SET_NAME=""

module load anaconda/2025
source activate myenv

python scripts/run_downscaling_patternscale.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --set-name "$SET_NAME"
