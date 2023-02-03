# This file is for generating rich dataset
import math
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

dataset = "cifar100"
num_clients = 50

maximum_sample = 60
minimum_sample = 10
maximum_num_label = 2
minimum_num_label = 2

def cluster(dataset, total_client):
    total_label = len(np.unique(dataset.targets))
    label_list = [i for i in range(total_label)]
    num_cluster = 5

    if num_cluster * (maximum_num_label + minimum_num_label)/2 < total_label:
        num_cluster = math.ceil(total_label/minimum_num_label)
    total_sample = len(dataset)
    
    labels = dataset.targets
    idxs = range(total_sample)
    idxs_labels = np.vstack((idxs, labels)).T
    
    dict_client = {}
    client_labels = []
    
    list_samples = np.random.multinomial(total_label, [1/10.]*10, size=10).tolist()
    # >>> np.random.multinomial(20, [1/6.]*6, size=2)
    # array([[3, 4, 3, 3, 4, 3],
    #    [2, 4, 3, 4, 0, 7]])
    print(list_samples)
    for list_sample in list_samples:
        for id in list_sample:
            label_per_client = id
            if len(label_list) >= label_per_client:
                this_set = np.random.choice(label_list, label_per_client, replace=False)
                print("Cover label", this_set)
                client_labels.append(list(this_set))
                label_list = list(set(label_list) - set(this_set))
            elif 0 < len(label_list) < label_per_client:
                remain = label_list.copy()
                # add_on = label_per_client - len(this_set)
                add_on = label_per_client - len(remain)
                label_list = [i for i in range(total_label) if i not in label_list]
                this_set = np.random.choice(label_list, add_on, replace=False)
                print("Cover label", remain + list(this_set))
                client_labels.append(remain + list(this_set))
                label_list = list(set(label_list) - set(remain + list(this_set)))
            else:
                label_list = [i for i in range(total_label)]
                this_set = np.random.choice(label_list, label_per_client, replace=False)
                print("Cover label", this_set)
                client_labels.append(list(this_set))
                label_list = list(set(label_list) - set(this_set))
    print(client_labels)
    
    num_client_added = len(client_labels)
    # for idx in range(num_client_added, total_client):
    #     client_labels.append(client_labels[idx % 5])
    
    for client_idx, client_labels in zip(range(total_client), client_labels):
        for label in client_labels:
            sample_per_label = np.random.randint(minimum_sample, maximum_sample)
            idxes = idxs_labels[idxs_labels[:,1] == label][:,0]
            if sample_per_label < len(idxes):
                label_idxes = np.random.choice(idxes, sample_per_label, replace=False)
            else:
                label_idxes = idxes
            if client_idx not in dict_client.keys():
                dict_client[client_idx] = label_idxes.tolist()
            else:
                dict_client[client_idx] += label_idxes.tolist()
            idxs_labels[label_idxes] -= 100
        
    return dict_client, total_label

output, total_labels = cluster(train_dataset, num_clients)

# Produce json file
dir_path = f"./dataset_idx/{dataset}/cluster_sparse/{num_clients}client/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
json.dump(output, open(dir_path + f"{dataset}_cluster_sparse.json", "w"), indent=4, cls=NpEncoder)
print("Output generated successfully")

# Produce stat file
stat = np.zeros([num_clients, total_labels])
for client_id, sample_idexes in output.items():
    for sample_id in sample_idexes:
        label = train_dataset.targets[int(sample_id)]
        stat[int(client_id), label] += 1

np.savetxt(dir_path + f"{dataset}_cluster_sparse_stat.csv", stat, delimiter=",", fmt="%d")
print("Stats generated successfully")