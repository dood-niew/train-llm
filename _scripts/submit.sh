#!/bin/bash
#SBATCH -p gpu                        # Specify partition [Compute/Memory/GPU]
#SBATCH -N 16 -c 64
#SBATCH --ntasks-per-node=1             # Specify number of tasks per node
#SBATCH -t 24:00:00                     # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200258                     # Specify project name
#SBATCH --gpus-per-node=4    
#SBATCH -J qwen2_finetune               # Specify job name
#SBATCH --output=../log/%j.out
#SBATCH -w lanta-g-[100-115]

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
export TORCH_EXTENSIONS_DIR=../.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:40960
export WANDB_MODE="offline"
# add here
export CUDA_HOME=/usr/local/cuda

######################
### load Module ###
######################
module restore
module load Mamba
module load PrgEnv-gnu
module load cpe-cuda/23.09
module load gcc/10.3.0
module load cudatoolkit/23.3_11.8
# module load aws-ofi-nccl

######################
### Set enviroment ###
######################
conda deactivate
conda activate /project/lt200258-aithai/llm/env-list/llama-factory5

python -c "import torch; print(f'Torch version: {torch.__version__}')"
######################
#### Set network #####
######################
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "Number of nodes: $COUNT_NODE"
######################

LOG_DIR="../log/${SLURM_JOB_ID}"
mkdir -p $LOG_DIR/node_log
export LOG_DIR=$LOG_DIR

srun --output=${LOG_DIR}/node_log/node-%t.out sh smultinode.sh