#!/bin/bash
#SBATCH --job-name=sib200_eval
#SBATCH --output=logs/sib200_eval_%j.out
#SBATCH --error=logs/sib200_eval_%j.err
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:a100:1
#SBATCH --account=your_project_account

start_time=$(date +%s)
echo "Job started at: $(date)"

# Load conda environment
source /path/to/conda/etc/profile.d/conda.sh
conda activate your_environment_name

# List of other models to evaluate (commented out for reference)
MODEL_IDS=(
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3.1-8B"
    "google/gemma-7b"
    "google/gemma-2-9b"
    "Qwen/Qwen2-7B"
    "Qwen/Qwen1.5-7B"
)

DATASET_PATH="dataset/path"
RESULTS_DIR="results/path"
DEVICE_INPUT="cuda:0"
NUM_EXAMPLES=3
CONFIG_FILE="./sib200_langs.txt"
CONFIGS=()


if [[ -f "$CONFIG_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ ! -z "$line" ]]; then
            CONFIGS+=("$line")
        fi
    done < "$CONFIG_FILE"
else
    echo "Config file not found!"
    exit 1
fi

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No configs found in $CONFIG_FILE"
    exit 1
fi

for MODEL_ID in "${MODEL_IDS[@]}"
do
    echo "Evaluate model: $MODEL_ID"
    CUDA_VISIBLE_DEVICES=0,1 python ./eval_sib200.py \
        --configs "${CONFIGS[@]}" \
        --dataset_base_path "$DATASET_PATH" \
        --results_dir "$RESULTS_DIR" \
        --model_id "$MODEL_ID" \
        --device "$DEVICE_INPUT" \
        --num_examples "$NUM_EXAMPLES" \
        --seed 42
done

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"

# Calculate and display the duration
duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"