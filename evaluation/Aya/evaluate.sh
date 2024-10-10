#!/bin/bash
#SBATCH --job-name=aya_metrics
#SBATCH --output=logs/aya_metrics_%j.out
#SBATCH --error=logs/aya_metrics_%j.err
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --account=your_project_account

# Record start time
start_time=$(date +%s)
echo "Job started at: $(date)"

# Load conda environment
source /path/to/conda/etc/profile.d/conda.sh
conda activate your_evaluation_environment

# List of models to evaluate
MODEL_IDS=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-7b-chat-hf"
    # ... (other model IDs)
)

# Evaluate each model
for MODEL_ID in "${MODEL_IDS[@]}"; do
    echo "Evaluating model: $MODEL_ID"
    python ./evaluate.py \
        --input_dir "/path/to/input/directory" \
        --output_dir "/path/to/output/directory" \
        --model_id "$MODEL_ID" \
        --task_name "Aya"
done

# Calculate and display job duration
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"