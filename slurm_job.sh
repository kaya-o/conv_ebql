#!/bin/bash
#SBATCH --job-name=ebql_k10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=All
#SBATCH --export=ALL
#SBATCH -D ./
#SBATCH --get-user-env

# Activate the virtual environment (create it first if it does not exist)
if [ -d .venv ]; then
    source .venv/bin/activate
else
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

pip install -r requirements.txt

mkdir -p logs

echo "[EBQL] Running with K=${K_VALUE}, run name=${RUN_NAME}, seeds=${SEEDS}"

python ebql.py \
    --run-name "ebql_k10" \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --total-steps 2000000 \
    --eval-every 100000 \
    --eval-episodes 20 \
    --K 10
