#!/bin/bash -l
# MdC submit script
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH --partition=troisi
#SBATCH --job-name=ML_d_a
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --time=120:00:00
export OMP_NUM_THREADS=1

#module load apps/anaconda3/2019.10-bioconda
#module load apps/intel-python/2019-3.6.8
#conda create -n myenv tornado
module load apps/anaconda3/2019.10-general
conda create -n myenv tornado
conda activate myenv
#conda activate myenv

echo "---------------------------------"
echo " "
echo "Job id: ${SLURM_JOBID}."
echo "Job name: ${SLURM_JOB_NAME}"
echo "Job submitted from:${SLURM_SUBMIT_DIR}"
echo " Using ${SLURM_JOB_NUM_NODES} node(s): ${SLURM_JOB_NODELIST}"
date
echo " "
echo "---------------------------------"

./NonFullereneAcceptorPrediction.py

echo " "
echo "Done!"
date
echo " "
echo "---------------------------------"
