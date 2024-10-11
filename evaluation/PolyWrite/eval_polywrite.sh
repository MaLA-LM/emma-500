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

# List of other models to evaluate (commented out for reference)

MODEL_ID="meta-llama/Llama-2-7b-hf"



CONFIG_FILE="./polywrite_langs.txt"
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

echo "Evaluate model: $MODEL_ID"

python ./eval_polywrite.py \
    --langs "${CONFIGS[@]}" \
    --dataset_base_path "/dataset/path" \
    --results_dir "./" \
    --model_id "$MODEL_ID" \
    --tensor_parallel_size 1 \
    --max_num_seqs 8 \
    --dtype 'auto' \

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"