#!/bin/bash
#SBATCH --job-name=aya_eval
#SBATCH --output=logs/aya_eval_%j.out
#SBATCH --error=logs/aya_eval_%j.err
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

# Define the model to evaluate
MODEL_ID="path/to/your/model/checkpoint"

# List of other models to evaluate (commented out for reference)
# MODEL_ID="meta-llama/Llama-2-7b-hf"
# ... (other model IDs)

# Run the evaluation script
python ./eval_aya.py \
    --langs "$(cat aya_langs.txt)" \
    --dataset_base_path "/path/to/dataset/Aya" \
    --results_dir "./Outputs" \
    --model_id "$MODEL_ID" \
    --tensor_parallel_size 1 \
    --max_num_seqs 16 \
    --dtype 'auto'

# Record end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"