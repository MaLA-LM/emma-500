#!/bin/bash
#SBATCH --job-name=polywrite_eval
#SBATCH --output=logs/polywrite_eval_%j.out
#SBATCH --error=logs/polywrite_eval_%j.err
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

MODEL_IDS=(
    "meta-llama/Llama-2-7b-hf"
)


for MODEL_ID in "${MODEL_IDS[@]}"; do
    echo "Evaluating model: $MODEL_ID"
    python ./evaluate.py \
        --input_dir "PolyWrite/Outputs" \
        --output_dir "PolyWrite/results"\
        --model_id "$MODEL_ID" \
        --task_name "PolyWrite"
done

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"