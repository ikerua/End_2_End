#!/bin/bash
#SBATCH --job-name=EVAL_WHISPER
#SBATCH --output=eval_modelos_%j.out
#SBATCH --error=eval_modelos_%j.err
#SBATCH --account=ehpc485
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# 1. Carga de modulos
module purge
module load intel
module load mkl
module load impi
module load hdf5
module load python/3.12.1
module load nasm
module load ffmpeg/7.1_dynamic

# 2. Entorno y PYTHONPATH
unset PYTHONPATH
export PYTHONPATH="/gpfs/projects/ehpc485/tesi681824/transcriptor/venv_transcriptor/lib/python3.12/site-packages"
source /gpfs/projects/ehpc485/tesi681824/transcriptor/venv_transcriptor/bin/activate

# 3. Restricciones de red
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "----------------------------------------------------------------"
echo "INICIO TRABAJO SLURM: EVALUACION COMPARATIVA DE 5 MODELOS"
date
echo "----------------------------------------------------------------"

DIR_PROYECTO="/gpfs/projects/ehpc485/tesi681824/transcriptor"
cd $DIR_PROYECTO

# Lanzar evaluacion
srun python -u test.py

echo "----------------------------------------------------------------"
echo "EVALUACION FINALIZADA"
date
echo "----------------------------------------------------------------"