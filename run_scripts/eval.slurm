#!/bin/bash
#SBATCH --job-name=smoke_test             # Job name
#SBATCH --partition=a40               # Target A40S node
#SBATCH --qos=a40                  
#SBATCH --gres=gpu:1                           # Request 1 GPU
#SBATCH --ntasks=1                             # No. of tasks 
#SBATCH --cpus-per-task=8                      # Max 8 CPUs per GPU
#SBATCH --mem=64G                              # RAM request
#SBATCH --time=3-00:00:00                      # Runtime (HH:MM:SS) 
#SBATCH --output=logs/smoke_test_%j.out   # Stdout log
#SBATCH --error=logs/smoke_test_%j.err

unset PYTHONPATH
unset PYTHONHOME
unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE

export VLLM_USE_FLASHINFER_SAMPLER=0
rm -rf ~/.cache/flashinfer/0.6.11.post2/86/cached_ops/sampling

# =========================================
# Activate Conda
# =========================================
# 1. Load environment
source ~/anaconda3/etc/profile.d/conda.sh

# =========================================
# Move to Project Directory
# =========================================

cd path/to/directory
set +e   # Don't exit on failure

# Diagnostics
hostname
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

echo "========================================="
echo "Starting experiments"
echo "========================================="
echo "Start time: $(date +"%Y-%m-%d %H:%M:%S")"


CONFIGS=(
    configs/eval_config.yaml
)

run_job() {

    CONFIG=$1
    NAME=$(basename "$CONFIG" .yaml)

    echo "========================================="
    echo "Running $NAME"
    echo "========================================="

    conda activate vllm

    echo "Starting vLLM server..."

    python -m vllm.entrypoints.openai.api_server \
        --model checkpoint_path \
        --served-model-name qwen-3.5-4b  \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len 33000 \
        --language-model-only \
        --port 8000 \
        > logs/vllm_server_0.log 2>&1 &

    VLLM_PID=$!

    echo "vLLM PID: $VLLM_PID"

    # Wait for server
    echo "Waiting for vLLM server..."

    until curl -s http://localhost:8000/v1/models >/dev/null; do
        sleep 10

        # Server died before becoming ready
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "vLLM failed to start for $NAME"
            return 1
        fi
    done

    echo "vLLM server is up!"

    # =====================================================
    # Run evaluation
    # =====================================================

    echo "Running evaluation..."

    python evaluation/run_eval.py --config "$CONFIG"
    JOB_STATUS=$?

    echo "Evaluation finished with exit code $JOB_STATUS"

    echo "Stopping vLLM..."

    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null

    if [ $JOB_STATUS -eq 0 ]; then
        echo "$NAME completed successfully."
    else
        echo "$NAME failed with exit code $JOB_STATUS"
    fi

    return $JOB_STATUS
}

# =========================================================
# Run all configs sequentially
# =========================================================
FAILED=()

for CONFIG in "${CONFIGS[@]}"; do
    run_job "$CONFIG"

    if [ $? -ne 0 ]; then
        FAILED+=("$CONFIG")
    fi

    echo "Waiting 60 seconds before the next job..."
    sleep 60
done

echo "========================================="
echo "All experiments finished."
echo "End time: $(date +"%Y-%m-%d %H:%M:%S")"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All jobs succeeded."
else
    echo "The following jobs failed:"
    printf '  %s\n' "${FAILED[@]}"
fi