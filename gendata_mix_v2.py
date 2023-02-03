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

train_dataset = datasets.MNIST( "./benchmark/mnist/data", train=True, download=True, transform=None)
print("total sample of dataset", len(train_dataset))

#-------- Change config here----------
num_clients = 50
maximum_num_label = 3
minimum_num_label = 2

rate = 0.7
#-------------------------------------

def cluster(dataset, total_client):
    total_label = len(np.unique(dataset.targets))
    label_list = [i for i in range(total_label)]

    # if num_cluster * (maximum_num_label + minimum_num_label)/2 < total_label:
    #     num_cluster = math.ceil(total_label/minimum_num_label)
    
    client_labels = []
    
    pivot = int(total_client * rate)
    
    for _ in range(pivot):
        label_per_client = np.random.randint(minimum_num_label, maximum_num_label+1)
        # label_per_client = 2
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
    
    for _ in range(pivot, total_client):
        client_labels.append(label_list)
    print(client_labels)
    return client_labels
    
def generate(dataset, list_client_labels):    
    total_label = len(np.unique(dataset.targets))
    total_sample = len(dataset)
    
    labels = dataset.targets
    idxs = range(total_sample)
    idxs_labels = np.vstack((idxs, labels)).T
    
    dict_client = {}
    
    position = int(num_clients*rate)
    
    
    for client_idx, client_labels in enumerate(list_client_labels):
        for label in client_labels:
            if(client_idx < position):
                minimum_sample = 20
                maximum_sample = 80
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

client_labels = cluster(train_dataset, num_clients) 
output, total_labels = generate(train_dataset, client_labels)                     

# Produce json file
dir_path = f"./dataset_idx/mnist/sparse{rate}_dense{round((1.0-rate),1)}/{num_clients}client/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
json.dump(output, open(dir_path + f"mnist_sparse{rate}_dense{round((1.0-rate),1)}.json", "w"), indent=4, cls=NpEncoder)
print("Output generated successfully")

# Produce stat file
stat = np.zeros([num_clients, total_labels])
for client_id, sample_idexes in output.items():
    for sample_id in sample_idexes:
        label = train_dataset.targets[int(sample_id)]
        stat[int(client_id), label] += 1

np.savetxt(dir_path + f"mnist_sparse{rate}_dense{round((1.0-rate),1)}_stat.csv", stat, delimiter=",", fmt="%d")
print("Stats generated successfully")