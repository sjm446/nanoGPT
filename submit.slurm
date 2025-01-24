#!/bin/bash
#SBATCH -p h100
#SBATCH --nodes=2

export OMP_NUM_THREADS=1
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=`cat /etc/hosts| tail -n 1 | awk '{print $1}'`
export NCCL_DEBUG=INFO

source nanogpt_venv/bin/activate

cd nanoGPT/
echo Run commands:
srun -N 1 python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=2  --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py config/train_gpt2.py &
srun -N 1 python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=2  --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py config/train_gpt2.py
