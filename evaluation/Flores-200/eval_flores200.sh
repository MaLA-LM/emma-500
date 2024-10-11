#!/bin/bash
#SBATCH --job-name=flores200_eval
#SBATCH --output=logs/flores200_eval_%j.out
#SBATCH --error=logs/flores200_eval_%j.err
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
    "meta-llama/Llama-2-7b-chat-hf"
    # ... (other model IDs)
)

TGT_LANG="eng_Latn"  # either "eng_Latn" for 'X-eng' translation, or "all" for 'eng-X' translation
N_shots=3

CONFIG_FILE="./flores200_langs.txt"
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
    python ./eval_flores200.py \
        --langs "${CONFIGS[@]}" \
        --tgt_lang "$TGT_LANG"\
        --dataset_base_path "/path/to/dataset/flores200" \
        --results_dir "./Translations/X-Eng-3shot" \
        --tgt_lang "$TGT_LANG" \
        --model_id "$MODEL_ID" \
        --nshots "$N_shots" \
        --tensor_parallel_size 1 \
        --max_num_seqs 32 \
        --dtype 'auto' \
        --seed 42
done

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"