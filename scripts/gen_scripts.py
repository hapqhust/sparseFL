import os

visible_cudas = [0, 1]
cudas = ",".join([str(i) for i in visible_cudas])
task_file = "main.py"

dataset = "cifar100"
# dataset_types = ["sparse_dir0.1_U5", "sparse_dir0.1_U10", "sparse_dir0.1_U15", "sparse_dir0.1_U20", "sparse_dir0.1_U40", "sparse_dir0.1_U60", "sparse_dir0.1_U80", "sparse_dir0.1_U100"]
# dataset_types = ["sparse_dir0.5_U80", "sparse_dir0.5_U100"]
dataset_types = ["sparse_dir0.2_U20", "sparse_dir0.3_U20", "sparse_dir0.4_U20"]

# config parameters
N = 100
rate = 0.1
K = int(N*rate)

E = 8
batch_size = 8
num_round = 1000

model = "resnet9"
# model = "cnn"

# algos = ["singleset"]
algos = ["singleset", "scaffold", "scaffold_with_random", "mp_fedavg", "fedavg_mp_with_random", "mp_fedprox", "mp_fedprox_with_random", "fedfa", "fedfa_with_random", "fedfv", "fedfv_with_random"]

data_folder = f"./benchmark/{dataset}/data"
log_folder = f"motiv/{dataset}"


header_text = "\
#!/bin/bash\n\
#$ -cwd\n\
#$ -l rt_G.small=1\n\
#$ -l h_rt=36:00:00\n\
#$ -o /home/aaa10078nj/Federated_Learning/Ha_SparseFL/logs/cifar100/$JOB_NAME_$JOB_ID.log\n\
#$ -j y\n\n\
source /etc/profile.d/modules.sh\n\
module load gcc/11.2.0\n\
module load openmpi/4.1.3\n\
module load cuda/11.5/11.5.2\n\
module load cudnn/8.3/8.3.3\n\
module load nccl/2.11/2.11.4-1\n\
module load python/3.10/3.10.4\n\
source ~/venv/pytorch1.11+horovod/bin/activate\n\n\
LOG_DIR=\"/home/aaa10078nj/Federated_Learning/Ha_SparseFL/logs/cifar100/$JOB_NAME_$JOB_ID\"\n\
rm -r ${LOG_DIR}\n\
mkdir ${LOG_DIR}\n\n\
#Dataset\n\
DATA_DIR=\"$SGE_LOCALDIR/$JOB_ID/\"\n\
cp -r ./sparseFL/benchmark/cifar100/data ${DATA_DIR}\n\n\
"

for dataset_type in dataset_types:
    
    task_name = f"{dataset}_{dataset_type}_N{N}_K{K}_E{E}"
    
    for algo in algos:
    
        command = f"\
        GROUP=\"{task_name}\"\n\
        ALG=\"{algo}\"\n\
        MODEL=\"{model}\"\n\
        WANDB=1\n\
        ROUND={num_round}\n\
        EPOCH_PER_ROUND={E}\n\
        BATCH={batch_size}\n\
        PROPOTION={rate}\n\
        NUM_THRESH_PER_GPU=1\n\
        NUM_GPUS=1\n\
        SERVER_GPU_ID=0\n\
        TASK=\"{task_name}\"\n\
        DATA_IDX_FILE=\"{dataset}/{dataset_type}/{N}client/{dataset}_sparse.json\"\n\n\
        cd sparseFL\n\n\
        "

        # task_name = f"{dataset}_{dataset_type}_N{N}_K{K}_E{E}"
        # command = formated_command.format(
        #     task_name, algo, model, 1000, E, batch_size, K/N, task_name, dataset_type, N
        # )
            
        body_text = "python main.py  --task ${TASK}  --model ${MODEL}  --algorithm ${ALG}  --wandb ${WANDB} --data_folder ${DATA_DIR}  --log_folder ${LOG_DIR}   --dataidx_filename ${DATA_IDX_FILE}   --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --num_threads_per_gpu ${NUM_THRESH_PER_GPU}  --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID} "

        dir_path = f"./run6/{dataset}/{dataset_type}/"
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
        file = open(dir_path + f"{task_name}_{algo}.sh", "w")
        file.write(header_text + command + body_text)
        file.close()
