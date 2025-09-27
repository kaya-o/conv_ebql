#!/bin/bash
#SBATCH --job-name=ebql_k
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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

K_VALUE=${K:-5}
RUN_NAME=${RUN_NAME:-spaceinvaders_k${K_VALUE}_full}
SEEDS=${SEEDS:-"0 1 2 3 4"}

echo "[EBQL] Running with K=${K_VALUE}, run name=${RUN_NAME}, seeds=${SEEDS}"

python ebql.py \
    --run-name "${RUN_NAME}" \
    --seeds ${SEEDS} \
    --total-steps 2000000 \
    --eval-every 100000 \
    --eval-episodes 20 \
    --K "${K_VALUE}"
