#!/usr/bin/env bash
set -e

RUN_DIR="logs/rsl_rl/sprint_2m/2026-11-03_18-48-27_sprint-scale-2048-1000"

cd /home/atuls/projects/microduck-challenges

for CKPT in 250 500 750 999; do
    echo "=================================================="
    echo ">>> Exporting Checkpoint ${CKPT} ..."
    echo "=================================================="
    .venv/bin/publish \
        --run "${RUN_DIR}" \
        --checkpoint "${CKPT}" \
        --repo "local/eval-${CKPT}" \
        --dry-run \
        --allow-dirty \
        --device cpu

    echo "=================================================="
    echo ">>> Scoring Checkpoint ${CKPT} on Arena ..."
    echo "=================================================="
    .venv/bin/python -m arena.cli score "publish-eval-${CKPT}/policy.onnx" --event sprint-2m | tee "publish-eval-${CKPT}/eval_result.txt"
done

echo "All evaluations complete!"
