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

num_clients = 200

maximum_sample = 12
minimum_sample = 6
maximum_num_label = 6
minimum_num_label = 3

def cluster(dataset, total_client):
    total_label = len(np.unique(dataset.targets))
    label_list = [i for i in range(total_label)]
    num_cluster = 10

    if num_cluster * (maximum_num_label + minimum_num_label)/2 < total_label:
        num_cluster = math.ceil(total_label/minimum_num_label)

    total_sample = len(dataset)
    
    labels = dataset.targets
    idxs = range(total_sample)
    idxs_labels = np.vstack((idxs, labels)).T
    
    dict_client = {}
    client_labels = []
    
    for _ in range(num_cluster):
        label_per_client = np.random.randint(minimum_num_label, maximum_num_label)
        if len(label_list) >= label_per_client:
            this_set = np.random.choice(label_list, label_per_client, replace=False)
            print("Cover label", this_set)
            client_labels.append(list(this_set))
            label_list = list(set(label_list) - set(this_set))
        elif 0 <= len(label_list) < label_per_client:
            remain = label_list.copy()
            add_on = label_per_client - len(this_set)
            label_list = [i for i in range(total_label) if i not in label_list]
            this_set = np.random.choice(label_list, label_per_client - add_on, replace=False)
            print("Cover label", remain + list(this_set))
            client_labels.append(remain + list(this_set))
            label_list = list(set(label_list) - set(remain + list(this_set)))
        else:
            print(len(label_list), label_per_client)

    num_added_total = (total_client - len(client_labels))
    num_added = np.random.default_rng().normal(total_client/2, total_client/5, len(client_labels))
    num_added = np.round(num_added/np.sum(num_added) * num_added_total)
    num_added = np.sort(num_added)
    num_added[-1] = num_added_total - np.sum(num_added) + num_added[-1]
    
    adds = []
    for i in range(len(client_labels)):
        for j in range(int(num_added[i])):
            adds += [client_labels[i]]
        
    client_labels += adds
    
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
dir_path = f"./dataset_idx/cifar100/cluster_sparse/{num_clients}client/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
json.dump(output, open(dir_path + "cifar100_sparse.json", "w"), indent=4, cls=NpEncoder)
print("Output generated successfully")

# Produce stat file
stat = np.zeros([num_clients, total_labels])
for client_id, sample_idexes in output.items():
    for sample_id in sample_idexes:
        label = train_dataset.targets[int(sample_id)]
        stat[int(client_id), label] += 1

np.savetxt(dir_path + "cifar100_sparse_stat.csv", stat, delimiter=",", fmt="%d")
print("Stats generated successfully")