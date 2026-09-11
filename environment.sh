# Source before eval.sh:  source environment.sh

# Required
export GITHUB_USERNAME=<SET>  # GitHub user for authenticated sparse clones of student CS6013 repos
export GITHUB_TOKEN=<SET>  # GitHub PAT used in clone URLs (keep private)
export SLACK_WEBHOOK_URL=<SET if you want to get slack alerts>  # Incoming webhook for format-check and per-roll eval alerts
export WEEK=<SET>  # Course week number; gates GitHub/HF paths as Week04

# Modes
export INTERACT=0  # 1 = stop on first failure and keep eval_workdir artifacts; 0 = continue and clean up
export SLACK_LOG=0  # 1 = attach Err/Traceback greps from eval/vLLM logs to Slack

# Paths / eval
export WORK_ROOT=  # Root for clones/checkpoints; empty → ./eval_workdir
export CONFIG=  # Eval YAML; empty → configs/eval_config.yaml
export DATASET=datasets/CS6013_sample_math_format_dataset  # Math dataset path (or HF id) passed to run_eval.py
export LIMIT=15  # Max number of eval examples
export MAX_CONCURRENCY=30  # In-flight chat requests against the vLLM server
export MAX_NEW_TOKENS=32000  # Generation budget; also sets max-model-len = 1000 + this

# Decompress / vLLM
export VLLM_GPU_MEMORY_UTILIZATION=0.35  # Fraction of total GPU memory vLLM may reserve
export VLLM_MAX_NUM_SEQS=50  # Max concurrent sequences (must fit Mamba cache at this utilization)
export VLLM_USE_FLASHINFER_SAMPLER=0  # 0 = skip FlashInfer sampler JIT (avoids ninja/build issues)

# GPU (read by CUDA / vLLM, not eval.sh itself)
export CUDA_VISIBLE_DEVICES=2  # Which physical GPU(s) vLLM can see
