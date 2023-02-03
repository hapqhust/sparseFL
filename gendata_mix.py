from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import numpy as np
import os
import math

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

training_data = datasets.MNIST(
    root="./benchmark/mnist/data/",
    train=True,
    download=False,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
)

testing_data = datasets.MNIST(
    root="./benchmark/mnist/data/",
    train=False,
    download=False,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
)

total_labels = np.unique(training_data.targets).tolist()
len(total_labels)
print(total_labels)

min_label_per_client = 2
max_label_per_client = 4

min_sample_per_client = 1
max_sample_per_client = 200

# num_clients = 5 * len(total_labels)
num_clients_sparse = 200
num_clients_dense = 0
num_clients = num_clients_sparse + num_clients_dense

total_label = len(total_labels)
label_list = [i for i in total_labels]
label_per_client = 2

labels = training_data.targets
idxs = range(len(training_data))
training_idxs_labels = np.vstack((idxs, labels)).T

labels = testing_data.targets
idxs = range(len(testing_data))
testing_idxs_labels = np.vstack((idxs, labels)).T

training_dict_client = {client_id:[] for client_id in range(num_clients)}
testing_dict_client = {client_id:[] for client_id in range(num_clients)}
# print(testing_idxs_labels)

client_labels = []
not_passed_label_list = label_list.copy()

for client_id in range(num_clients):
    label_per_client = np.random.randint(min_label_per_client, max_label_per_client + 1)
    this_set = np.random.choice(label_list, label_per_client, replace=False)
    client_labels.append(list(this_set))
    not_passed_label_list = list(set(not_passed_label_list) - set(this_set))

if len(not_passed_label_list) > 0:
    print("Uncover", len(not_passed_label_list), "labels !")
    exit(0)
else:
    print("Uncover", len(not_passed_label_list), "labels !")

samples_details = []

for client_idx, client_label in zip(range(num_clients), client_labels):
    sample_this_client = []
    
    print(f"---------------{client_idx}-------------")
    for label in client_label:
        print(f"Start with label: {client_idx}")
        sample_per_client = 0
        if (client_idx < num_clients_sparse):
          sample_per_client = np.random.randint(min_sample_per_client, int(max_sample_per_client/25))
        else:
          sample_per_client = np.random.randint(min_sample_per_client*100, max_sample_per_client + 1)
        sample_this_client.append(sample_per_client)
        print(f"Num of sample: {sample_per_client}")
        idxes_1 = training_idxs_labels[training_idxs_labels[:,1] == label][:,0]
        idxes_2 = testing_idxs_labels[testing_idxs_labels[:,1] == label][:,0]
        
        label_1_idxes = np.random.choice(idxes_1, sample_per_client, replace=False)
        label_2_idxes = np.random.choice(idxes_2, max(int(sample_per_client/4), 1), replace=False)
        
        training_dict_client[client_idx] += label_1_idxes.tolist()
        testing_dict_client[client_idx] += label_2_idxes.tolist()
        
        training_idxs_labels[label_1_idxes] -= 100
        testing_idxs_labels[label_2_idxes] -= 100
    
    samples_details.append(sample_this_client)


dis_mtx = np.zeros([num_clients, total_label])
for client_id in range(len(client_labels)):
    client_label = client_labels[client_id]
    client_samples = samples_details[client_id]
    
    for label, num_samples in zip(client_label, client_samples):
        dis_mtx[client_id][total_labels.index(label)] = num_samples
        
    
dataset = "mnist"
# type_dataset = "sparse3_dense7"
type_dataset = "sparse"
savepath = f"./dataset_idx/{dataset}/{type_dataset}/{num_clients}client"

if not Path(savepath).exists():
    os.makedirs(savepath)
    
json.dump(training_dict_client, open(f"{savepath}/{dataset}_{type_dataset}.json", "w"), cls=NumpyEncoder)
json.dump(testing_dict_client, open(f"{savepath}/{dataset}_{type_dataset}_test.json", "w"), cls=NumpyEncoder)
np.savetxt(f"{savepath}/{dataset}_{type_dataset}_stat.csv", dis_mtx, fmt="%d", delimiter=",")