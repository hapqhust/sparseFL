mkdir fedtask

ROUND=400
EPOCH_PER_ROUND=8
PROPOTION=0.2
NUM_CLIENT=30
K=6

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_cluster_sparse_fedfv_N${NUM_CLIENT}_K${K} --wandb 1 --model resnet9 --algorithm fedfv_with_random --data_folder ./benchmark/cifar10/data --log_folder fedtask --dataidx_filename cifar10/cluster_sparse/${NUM_CLIENT}client/cifar10_cluster_sparse.json --num_rounds ${ROUND}  --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size 10 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_cluster_sparse_fedfv_N${NUM_CLIENT}_K${K} --wandb 1 --model resnet9 --algorithm fedfv --data_folder ./benchmark/cifar10/data --log_folder fedtask --dataidx_filename cifar10/cluster_sparse/${NUM_CLIENT}client/cifar10_cluster_sparse.json --num_rounds ${ROUND}  --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size 10 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0