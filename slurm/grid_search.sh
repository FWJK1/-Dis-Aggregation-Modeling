#!/bin/bash -l

#SBATCH --job-name=mocs_grid
#SBATCH --array=0-197

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# 10 min/task observed; 30 min cap for safety
#SBATCH --time=00:30:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fkeenank@uvm.edu

#SBATCH --output=/users/f/k/fkeenank/logs/mocs_grid-%A_%a.out
#SBATCH --error=/users/f/k/fkeenank/logs/mocs_grid-%A_%a.err

set -e

module purge
module load miniforge 
source activate data-science

my_job_header

echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Working directory: $(pwd)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

cd /users/f/k/fkeenank/Disaggregation
python -u -m src.run --task_id $SLURM_ARRAY_TASK_ID

echo "End time: $(date)"
