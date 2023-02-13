mkdir fedtask

ROUND=800
EPOCH_PER_ROUND=8
PROPOTION=0.2
NUM_CLIENT=50
K=10

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_cluster_sparse_singleset_N${NUM_CLIENT}_K${K} --wandb 1 --model resnet9 --algorithm singleset --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/cluster_sparse/${NUM_CLIENT}client/cifar100_cluster_sparse.json --num_rounds ${ROUND}  --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_cluster_sparse_singleset_N${NUM_CLIENT}_K${K} --wandb 0 --model resnet9 --algorithm singleset_with_random --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/cluster_sparse/${NUM_CLIENT}client/cifar100_cluster_sparse.json --num_rounds ${ROUND}  --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0