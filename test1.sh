mkdir fedtask

ROUND=500
EPOCH_PER_ROUND=8
PROPOTION=0.20

CUDA_VISIBLE_DEVICES=1,0 python main.py --task mnist_cluster_sparse_N50_K10_E8 --wandb 1 --model cnn --algorithm mp_proposal_cluster_matching_v2 --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/cluster_sparse/50client/mnist_cluster_sparse.json --num_rounds ${ROUND}  --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size 2 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1,0 python main.py --task mnist_cluster_sparse_N50_K10_E8 --wandb 1 --model cnn --algorithm mp_proposal_cluster_matching_v1 --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/cluster_sparse/50client/mnist_cluster_sparse.json --num_rounds ${ROUND}  --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size 2 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1,0 python main.py --task mnist_cluster_sparse_N50_K10_E8 --wandb 1 --model cnn --algorithm mp_proposal_random_matching --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/cluster_sparse/50client/mnist_cluster_sparse.json --num_rounds ${ROUND}  --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size 2 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0