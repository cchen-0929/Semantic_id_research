#!/bin/bash
#SBATCH --job-name=idgenrec_recdata_beauty
#SBATCH --account=bffn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hchen42@illinois.edu
#SBATCH --output=/u/hchen42/IDGenRec/logs/idgenrec_recdata_beauty_%j.out
#SBATCH --error=/u/hchen42/IDGenRec/logs/idgenrec_recdata_beauty_%j.err

set -euo pipefail

mkdir -p /u/hchen42/IDGenRec/logs /u/hchen42/IDGenRec/results

module purge
module load PrgEnv-gnu
module load cudatoolkit/25.3_12.8
module load pytorch-conda/2.8

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd /u/hchen42/IDGenRec/src

echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "Dataset: Beauty"
echo "================================================"

python - << 'PY'
import torch, transformers, sklearn, numpy
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("sklearn:", sklearn.__version__)
print("numpy:", numpy.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

python main_generative.py \
  --distributed 1 \
  --gpu 0 \
  --data_format recdata \
  --data_path /u/hchen42/Semantic_ID/RecData \
  --datasets Beauty \
  --tasks sequential \
  --item_indexing sequential \
  --prompt_file /u/hchen42/IDGenRec/prompt.txt \
  --test_prompt seen:0 \
  --topk 1,5,10,20,100 \
  --recdata_output_dir /u/hchen42/IDGenRec/results \
  --train 0 \
  --backbone t5-small

echo "================================================"
echo "IDGenRec RecData Beauty finished at: $(date)"
echo "================================================"

