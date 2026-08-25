#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

for seed in $(seq 1 30); do
    echo "========================================"
    echo "  seed $seed"
    echo "========================================"

    ./diablo-ai.py agent-ai \
        --embedding-dim 512 \
        --cnn-arch cnn32expert \
        --model ClearAllLevels \
        --env Diablo-ClearAllLevels-v17 \
        --dungeon-level 1 \
        --seed-base "$seed" \
        --max-steps-per-level 5000 \
        --kill-threshold 1 \
        --stat-strategy dex-rush

    echo "========================================"
    echo "  seed $seed done"
    echo "========================================"
done


#        --stat-strategy 1+=2s2d1v
