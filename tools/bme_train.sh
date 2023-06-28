#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --job-name=DINO_MIMIC
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 5-00:00:00

set -x

CONFIG=$1
PY_ARGS=${@:2}
source activate det
nvidia-smi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}