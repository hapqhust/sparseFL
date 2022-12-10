mkdir fedtask

CUDA_VISIBLE_DEVICES=1,0 python main.py --task mnist_cluster_sparse_N10_K10 --wandb 0 --model cnn --algorithm mp_proposal_4_v3_clustering --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/cluster_sparse/10client/mnist_sparse.json --num_rounds 50 --num_epochs 8 --proportion 1 --batch_size 2 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1,0 python main.py --task mnist_cluster_sparse_N10_K10 --wandb 0 --model cnn --algorithm mp_proposal_4 --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/cluster_sparse/10client/mnist_sparse.json --num_rounds 50 --num_epochs 4 --proportion 1 --batch_size 2 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --se 4
CUDA_VISIBLE_DEVICES=1,0 python main.py --task mnist_cluster_sparse_N10_K10 --wandb 0 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/cluster_sparse/10client/mnist_sparse.json --num_rounds 50 --num_epochs 8 --proportion 1 --batch_size 2 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
