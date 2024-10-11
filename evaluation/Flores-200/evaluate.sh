#!/bin/bash
#SBATCH --job-name=flores200_metrics
#SBATCH --output=logs/flores200_metrics_%j.out
#SBATCH --error=logs/flores200_metrics_%j.err
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

for MODEL_ID in "${MODEL_IDS[@]}"; do
    echo "Evaluating model: $MODEL_ID"
    python ./evaluate.py \
        --translations_dir "translations/X-Eng-3shot" \
        --output_dir "results/X-Eng-3"\
        --model_id "$MODEL_ID" \
        --task_name "Flores-200-X-Eng"
done

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"