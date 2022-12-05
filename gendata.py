# This file is for generating sparse dataset
from torchvision import datasets
import json
import os
from pathlib import Path
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

train_dataset = datasets.CIFAR100( "./benchmark/cifar100/data", train=True, download=True, transform=None)
print("total sample of dataset", len(train_dataset))

num_clients = 500
maximum_sample = 8
minimum_sample = 4

total_sample = len(train_dataset)
sample_indexes = [i for i in range(total_sample)]

# Produce data_idx file
output = {}
for client_i in range(num_clients):
    n_sample = np.random.randint(minimum_sample, maximum_sample)
    idxes = np.random.choice(sample_indexes, n_sample)
    output[str(client_i)] = list(idxes)

    sample_indexes = list(set(sample_indexes) - set(list(idxes)))

dir_path = f"./dataset_idx/cifar100/sparse/{num_clients}client/"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

json.dump(output, open(dir_path + "cifar100_sparse.json", "w"), indent=4, cls=NpEncoder)
print("Output generated successfully")
# Produce stat file

labels = train_dataset.targets
total_labels = np.unique(labels)

stat = np.zeros([num_clients, len(total_labels)])

for client_id, sample_idexes in output.items():
    for sample_id in sample_idexes:
        label = train_dataset.targets[int(sample_id)]
        stat[int(client_id), label] += 1

np.savetxt(dir_path + "cifar100_sparse_stat.csv", stat, delimiter=",", fmt="%d")
print("Stats generated successfully")