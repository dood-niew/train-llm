#!/bin/bash
#SBATCH -p compute-devel				# Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 128                    # Specify number of nodes and processors per task #SMATCH --mem 500GB 
#SBATCH --gpus-per-task=0           # Specify the number of GPUs
#SBATCH --ntasks-per-node=1         # Specify tasks per node
#SBATCH -t 2:00:00                  # Specify maximum time limit (hour: minute: second)d
#SBATCH -A lt200258                 # Specify project name
#SBATCH -J merge               # Specify job name
#SBATCH --output=../log/merge%j.out

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;

export HF_HOME=../.cache
export HF_HUB_CACHE=../.cache
export HF_DATASETS_CACHE=../.cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GPUS_PER_NODE=4

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_TIMEOUT=360000
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_EXTENSIONS_DIR=/project/lt200258-aithai/jack/playground/LLaMA-Factory/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:40960
export WANDB_MODE="offline"


######################
### load Module ###
######################
module restore
module load Mamba/23.11.0-0
module load PrgEnv-gnu
module load cpe-cuda/23.03
module load gcc/10.3.0
module load cudatoolkit/23.3_11.8

######################
### Set enviroment ###
######################
conda deactivate
conda activate /lustrefs/disk/project/lt200258-aithai/jack/playground/LLaMA-Factory/env

which python

######################
#### Set network #####
######################
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802 #12999 #12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "Number of nodes: $COUNT_NODE"
######################


python ./merge.py \
    --base_model_path "/project/lt200258-aithai/jack/playground/lora_finetune/output/apt3-test-30step-full" \
    --lora_model_path "/project/lt200258-aithai/jack/playground/lora_finetune/output/apt3-30steps-full-lora" \
    --output_directory "/project/lt200258-aithai/jack/playground/lora_finetune/output_merge/apt3-30steps-full-lora" \
    --config '{"max_position_embeddings": 131072, "eos_token_id": 151645}'