#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=36:00:00
#$ -o /home/aaa10078nj/Federated_Learning/Ha_SparseFL/logs/cifar100/$JOB_NAME_$JOB_ID.log
#$ -j y

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load openmpi/4.1.3
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
module load python/3.10/3.10.4
source ~/venv/pytorch1.11+horovod/bin/activate

LOG_DIR="/home/aaa10078nj/Federated_Learning/Ha_SparseFL/logs/cifar100/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}

#Dataset
DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
cp -r ./sparseFL/benchmark/cifar100/data ${DATA_DIR}

        GROUP="cifar100_sparse_dir100_N300_K30_E8"
        ALG="singleset"
        MODEL="resnet9"
        WANDB=1
        ROUND=1000
        EPOCH_PER_ROUND=8
        BATCH=8
        PROPOTION=0.1
        NUM_THRESH_PER_GPU=1
        NUM_GPUS=1
        SERVER_GPU_ID=0
        TASK="cifar100_sparse_dir100_N300_K30_E8"
        DATA_IDX_FILE="cifar100/sparse_dir100/300client/cifar100_sparse.json"

        cd sparseFL

        python main.py  --task ${TASK}  --model ${MODEL}  --algorithm ${ALG}  --wandb ${WANDB} --data_folder ${DATA_DIR}  --log_folder ${LOG_DIR}   --dataidx_filename ${DATA_IDX_FILE}   --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --num_threads_per_gpu ${NUM_THRESH_PER_GPU}  --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID} 