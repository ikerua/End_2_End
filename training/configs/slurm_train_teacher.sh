#!/bin/bash
#SBATCH --job-name=W_KD_EUS
#SBATCH --output=train_distillation_%j.out
#SBATCH --error=train_distillation_%j.err
#SBATCH --account=ehpc485
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
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

# 3. Candados Offline y Proteccion de Red (NCCL)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200000 

echo "----------------------------------------------------------------"
echo "INICIO TRABAJO SLURM: KNOWLEDGE DISTILLATION (Teacher-Student)"
date
echo "----------------------------------------------------------------"

# 4. Estrategia de Disco Local (SSD TMPDIR)
DIR_PROYECTO="/gpfs/projects/ehpc485/tesi681824/transcriptor"
ORIGIN_DATA="$DIR_PROYECTO/dataset_unificado"
DEST_DATA="$TMPDIR/dataset_unificado"

echo "Copiando dataset unificado al SSD del nodo ($TMPDIR)..."
cp -r $ORIGIN_DATA $TMPDIR/
echo "Copia finalizada."

# 5. Ejecutar Entrenamiento
cd $DIR_PROYECTO

echo "Lanzando PyTorch Lightning (Distillation)..."
srun python -u TeacherStudent/train_distillation.py \
    --data_dir $DEST_DATA \
    --teacher_model_path /gpfs/projects/ehpc485/tesi681824/transcriptor/modelo_whisper_large_v3 \
    --teacher_ckpt_path /gpfs/scratch/ehpc485/tesi681824/whisper_eu_out/checkpoints/last-v2.ckpt \
    --student_model_name /gpfs/projects/ehpc485/tesi681824/transcriptor/modelo_whisper_base \
    --num_nodes 1 \
    --gpus_per_node 4 \
    --num_workers 4 \
    --batch_size 4 \
    --accumulate_grad_batches 16 \
    --learning_rate 5e-5 \
    --temperature 2.0 \
    --alpha 0.5

echo "----------------------------------------------------------------"
echo "ENTRENAMIENTO FINALIZADO"
date
echo "----------------------------------------------------------------"