#!/usr/bin/env bash
#SBATCH --job-name=saa_matlab_all
#SBATCH -p hpg-default
#SBATCH --qos=yu.yang1-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH -t 24:00:00
#SBATCH --array=0-13
#SBATCH --output=matlab_out/%A_%a.out
#SBATCH --error=matlab_err/%A_%a.err

COMMANDS=(
  "lasso_ista.m"
  "mirror_descent_1point1_initialrandom.m"
  "mirror_descent_1point1.m"
  "mirror_descent_1point5.m"
  "mirror_descent_2.m"
  "mirror_descent_entropic.m"
  "SAA_varyingN_replication_1point01_new.m"
  "SAA_varyingN_replication_1point01_new_shrink.m"
  "SAA_varyingN_replication_1point5_new.m"
  "SAA_varyingN_replication_1point5_new_shrink.m"
  "SAA_varyingN_replication_2_new.m"
  "SAA_varyingN_replication_2_new_shrink.m"
  "SAA_varyingN_replication_initial0.m"
  "SAA_varyingN_replication_initialrandom.m"
)

JOB_SCRIPT=${COMMANDS[$SLURM_ARRAY_TASK_ID]}

mkdir -p matlab_out matlab_err

cd "$SLURM_SUBMIT_DIR"

if command -v module >/dev/null 2>&1; then
  module load matlab
fi

matlab -nodisplay -nosplash -nodesktop -r "try; run('${JOB_SCRIPT}'); catch ME; disp(getReport(ME)); exit(1); end; exit"
