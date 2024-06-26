#!/bin/bash -l
#SBATCH -J seir_dist150
#SBATCH --output=seir_dist150.out
#SBATCH --mail-user=damian.ndiwago@uni.lu
#SBATCH --mail-type=END,FAIL
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#BATCH --mem=1TB
#SBATCH -p batch
#SBATCH --qos long
#SBATCH --time=7-00:00:00
echo "== Starting run at $(date)"
echo "== Job name: ${SLURM_JOB_NAME}"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"
echo "== Number of tasks: ${SLURM_NTASKS}"

module load lang/Python/3.8.6-GCCcore-10.2.0
source $HOME/damian/damian-venv/bin/activate

# TODO: On more modern GNU parallel this makes 01, 02, ...,10
parallel -j1 --plus python3 $HOME/prior_impact/examples/SEIR/prior_samples/wasser_dist_seir150.py --output_dir $SCRATCH/thesis-ideas/hbv/study_two_3/{0#} ::: {1..1}

