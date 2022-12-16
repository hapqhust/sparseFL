import os

visible_cudas = [0, 1]
cudas = ",".join([str(i) for i in visible_cudas])
task_file = "main.py"

dataset = "cifar10"
noniid = "pareto"
N = 100
K = 10
total_epochs = 8000
batch_size = 16

model = "cnn"
algos = ["scaffold", "mp_proposal", "mp_fedavg"]
data_folder = f"./benchmark/{dataset}/data"
log_folder = f"motiv/{dataset}"


header_text = "\
#!/bin/bash\n\
#$ -cwd\n\
#$ -l rt_G.small=1\n\
#$ -l h_rt=36:00:00\n\
#$ -o /home/aaa10078nj/Federated_Learning/Hung_test/logs/motiv1/$JOB_NAME_$JOB_ID.log\n\
#$ -j y\n\n\
source /etc/profile.d/modules.sh\n\
module load gcc/11.2.0\n\
module load openmpi/4.1.3\n\
module load cuda/11.5/11.5.2\n\
module load cudnn/8.3/8.3.3\n\
module load nccl/2.11/2.11.4-1\n\
module load python/3.10/3.10.4\n\
source ~/venv/pytorch1.11+horovod/bin/activate\n\n\
LOG_DIR=\"/home/aaa10078nj/Federated_Learning/Hung_test/logs/motiv1/cifar10/$JOB_NAME_$JOB_ID\"\n\
rm -r ${LOG_DIR}\n\
mkdir ${LOG_DIR}\n\n\
#Dataset\n\
DATA_DIR=\"$SGE_LOCALDIR/$JOB_ID/\"\n\
cp -r ./easyFL/benchmark/cifar10/data ${DATA_DIR}\n\n\
"

formated_command = "\
GROUP=\"{}\"\n\
ALG=\"{}\"\n\
MODEL=\"{}\"\n\
WANDB=1\n\
ROUND={}\n\
EPOCH_PER_ROUND={}\n\
BATCH={}\n\
PROPOTION={:>.2f}\n\
NUM_THRESH_PER_GPU=1\n\
NUM_GPUS=1\n\
SERVER_GPU_ID=0\n\
TASK=\"{}\"\n\
DATA_IDX_FILE=\"cifar10/100client/CIFAR10_100client_pareto.csv\"\n\n\
cd easyFL\n\n\
"

for E in [1, 5, 10, 20, 25, 40, 50]:
    task_name = f"{dataset}_{noniid}_N{N}_K{K}_E{E}"

    for algo in algos:
        if E < 20:
            command = formated_command.format(
                task_name, algo, model, int(total_epochs/E), E, batch_size, K/N, task_name
            )
        else:
            command = formated_command.format(
                task_name, algo, model, 500, E, batch_size, K/N, task_name
            )
            
        body_text = "python main.py  --task ${TASK}  --model ${MODEL}  --algorithm ${ALG}  --wandb ${WANDB} --data_folder ${DATA_DIR}  --log_folder ${LOG_DIR}   --dataidx_filename ${DATA_IDX_FILE}   --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --num_threads_per_gpu ${NUM_THRESH_PER_GPU}  --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID} "

        file = open(f"./motiv1/{task_name}_{algo}.sh", "w")
        file.write(header_text + command + body_text)
        file.close()