#!/bin/bash
#SBATCH --job-name=eager_recdata_rest
#SBATCH --account=bffn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hchen42@illinois.edu
#SBATCH --output=/u/hchen42/EAGER/logs/eager_rest_%j.out
#SBATCH --error=/u/hchen42/EAGER/logs/eager_rest_%j.err

set -euo pipefail

mkdir -p /u/hchen42/EAGER/logs /u/hchen42/EAGER/results

module purge
module load PrgEnv-gnu
module load cudatoolkit/25.3_12.8
module load pytorch-conda/2.8

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd /u/hchen42/EAGER

DATASETS=("Sports_and_Outdoors" "Toys_and_Games" "Yelp")

echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "================================================"

for DATASET in "${DATASETS[@]}"; do
  echo "------------------------------------------------"
  echo "Running: ${DATASET} at $(date)"
  echo "------------------------------------------------"

  python run_recdata_eager.py \
    --data_path /u/hchen42/Semantic_ID/RecData \
    --dataset "${DATASET}" \
    --output_dir /u/hchen42/EAGER/results \
    --eager_root /u/hchen42/EAGER/EAGER \
    --seq_len 30 \
    --batch_size 256 \
    --epochs 12 \
    --lr 0.0005 \
    --d_model 128 \
    --n_head 4 \
    --enc_num_layers 2 \
    --dec_num_layers 2 \
    --k 256 \
    --max_iters 50 \
    --feature_ratio 1.0 \
    --select_metric NDCG \
    --select_k 10 \
    --patience 3 \
    --topk 1,5,10,20,100 \
    --seed 2026 \
    --gpu 0

  echo "Completed: ${DATASET}"
done

echo "================================================"
echo "All remaining datasets finished at: $(date)"
echo "================================================"
