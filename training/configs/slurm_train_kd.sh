#!/bin/bash
#SBATCH --job-name=whisper-kd
#SBATCH --output=logs/whisper_kd_%j.out
#SBATCH --error=logs/whisper_kd_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --exclusive

# ──────────────────────────────────────────────────────────────────────
#  SLURM job script for Knowledge Distillation on MareNostrum
#  Teacher: Whisper Large V3 (frozen) → Student: Whisper Small (trainable)
#  2 nodes × 4 GPUs = 8 GPUs total, FP32 precision
# ──────────────────────────────────────────────────────────────────────

# Load required modules (adjust to your MareNostrum environment)
# module load python/3.11
# module load cuda/12.x
# module load nccl

# Activate virtual environment
# source .venv/bin/activate

# ─── Multi-node communication setup ──────────────────────────────────
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))

echo "=================================================="
echo "  KNOWLEDGE DISTILLATION JOB"
echo "  JOB ID:       $SLURM_JOB_ID"
echo "  MASTER_ADDR:  $MASTER_ADDR"
echo "  MASTER_PORT:  $MASTER_PORT"
echo "  WORLD_SIZE:   $WORLD_SIZE"
echo "=================================================="

# ─── NCCL optimizations for MareNostrum Infiniband ────────────────────
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=INFO

# ─── Launch KD training ──────────────────────────────────────────────
srun python -m experiments.train_kd \
    --teacher_path ./checkpoints/whisper-euskera-final \
    --student_path openai/whisper-small \
    --data_dir_clean ./data/clean \
    --data_dir_noisy ./data/noisy \
    --use_clean_data \
    --temperature 2.0 \
    --alpha 0.7 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_epochs 50 \
    --max_steps 80000 \
    --accumulate_grad_batches 1 \
    --num_nodes 2 \
    --gpus_per_node 4 \
    --num_workers 8 \
    --output_dir ./checkpoints/kd \
    --log_dir ./logs
