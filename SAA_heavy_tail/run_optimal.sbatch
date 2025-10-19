#!/usr/bin/env bash
#SBATCH --job-name=saa_matlab_all
#SBATCH -p hpg-default
#SBATCH --qos=yu.yang1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH -t 1-12:00:00
#SBATCH --array=0-0
#SBATCH --output=matlab_out/%A_%a.out
#SBATCH --error=matlab_err/%A_%a.err

set -euo pipefail

COMMANDS=(
  "SAA_varyingN_optimal_solution_auto_matrix.m"
)

JOB_SCRIPT=${COMMANDS[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"

if command -v module >/dev/null 2>&1; then
  module load matlab
fi

matlab -nodisplay -nosplash -nodesktop -r "try; run('${JOB_SCRIPT}'); catch ME; disp(getReport(ME)); exit(1); end; exit"
