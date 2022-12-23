from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import copy
import torch
import numpy as np
import torch.multiprocessing as mp
import random
from sklearn.metrics.pairwise import cosine_similarity
import math


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.clusters = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        self.phase_one_models = None
        self.phase_one_ids = None

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        
        # print(self.selected_clients)
        # Stage 1
        self.phase_one_ids, self.phase_one_models, _ = self.communicate(
            selected_clients = self.selected_clients, pool = pool)
        self.phase_one_models = [i.to("cpu") for i in self.phase_one_models]
        base = copy.deepcopy(self.model).to("cpu")
        
        print(self.phase_one_ids)
        # self.phase_one_shuffle(t)

        clt = []

        for idx, model in zip(self.phase_one_ids, self.phase_one_models):
            trained_model = [param[:]
                             for name, param in model.named_parameters()]
            base_model = [param[:]
                          for name, param in base.named_parameters()]
            
            res = torch.sub(trained_model[-1], base_model[-1]).detach().numpy()
            # res2 = np.where(res < 0, res*0.1, res)
            clt.append(res)

        # clt = torch.FloatTensor(np.array(clt))
        data = np.asarray(clt, dtype=float)
        data = self.unit_scaler(data)
        N = len(data)
        label, num_clusters = self.classifier(data, N, threshold=1.2)
        
        print(label)
        # print(num_clusters)
        
        pairings = self.pairing_clients(
            clients=self.phase_one_ids, group_label=label, num_clusters=num_clusters, clients_per_group=2)

        print(pairings)
        # self.phase_one_models = [i.to("cpu") for i in self.phase_one_models]
        
        # Stage 2
        _, phase_two_models, _ = self.communicate(pairs = pairings, pool = pool)
        self.phase_one_models = None

        if not self.selected_clients:
            return

        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in phase_two_models]
        self.model = self.aggregate(models, p = [1.0 for i in range(len(models))])
        return

    def pairing_clients(self, clients, group_label, num_clusters, clients_per_group=2):
        # participants = clients.copy()
        pairs = []
        rest = []
        for i in range(num_clusters):
            group = np.where(group_label == i+1)
            participants = [clients[i] for i in group[0]]
            while len(participants) > 1:
                one_pair = list(np.random.choice(
                    participants, clients_per_group, replace=False))
                pairs.append(one_pair)
                participants = list(set(participants) - set(one_pair))
            if len(participants):
                pairs.append(participants)
        
        # while len(rest) > 1:
        #     # print(rest[0])
        #     one_pair = list(np.random.choice(rest, clients_per_group, replace=False))
        #     pairs.append(one_pair)
        #     rest = list(set(rest) - set(one_pair))
        # if len(rest):
        #     pairs.append(rest)
        return pairs

    def pack(self, client_id):
        if not self.phase_one_models:
            send_model = copy.deepcopy(self.model)
        else:
            send_model = self.phase_one_models[self.phase_one_ids.index(
                client_id)]
            print(client_id, self.phase_one_ids.index(client_id))
        
        return {
            "model": send_model,
        }

    def unpack(self, packages_received_from_clients):
        ids = [cp["id"] for cp in packages_received_from_clients]
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"]
                        for cp in packages_received_from_clients]
        return ids, models, train_losses

    def communicate(self, selected_clients = None, pairs = None, pool=None):
        packages_received_from_clients = []
        
        if selected_clients:
            packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
            self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        else:
            packages_received_from_clients = pool.map(
                self.communicate_with_pairs, pairs)
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        # sortlist = sorted(packages_received_from_clients,
        #                   key=lambda d: d['id'])
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, client_id):
        gpu_id = int(mp.current_process().name[-1]) - 1
        gpu_id = gpu_id % self.gpus

        torch.manual_seed(0)
        torch.cuda.set_device(gpu_id)
        # This is only 'cuda' so its can find the propriate cuda id to train
        device = torch.device('cuda')

        # package the necessary information
        svr_pkg = self.pack(client_id)
        # listen for the client's response and return None if the client drops out
        if self.clients[client_id].is_drop():
            return None
        return self.clients[client_id].reply(svr_pkg, device)
    
    def communicate_with_pairs(self, pairs):
        gpu_id = int(mp.current_process().name[-1]) - 1
        gpu_id = gpu_id % self.gpus

        torch.manual_seed(0)
        torch.cuda.set_device(gpu_id)
        # This is only 'cuda' so its can find the propriate cuda id to train
        device = torch.device('cuda')
        # print(pairs)
        model_clientid = pairs[0]
        train_clientid = pairs[1] if len(pairs) > 1 else pairs[0]
        svr_pkg = self.pack(client_id = model_clientid)
        if self.clients[train_clientid].is_drop():
            return None
        return self.clients[train_clientid].reply(svr_pkg, device)

    def unit_scaler(self, data):
        """
        convert length of vector into 1
        :param
            data: 
        """
        for id, vector in enumerate(data):
            length_of_vector = 0
            for i in range(vector.size):
                length_of_vector += pow(vector[i], 2)

            data[id, :] = data[id, :] / math.sqrt(length_of_vector)

        return data

    def classifier(self, data, N, threshold):
        # init
        label = np.zeros((N,), dtype=int)
        num_cluster = 1

        while True:
            # list index của nodes chưa được chia vào nhóm nào
            filtered_label = np.where(label == 0)[0]
            np.random.shuffle(filtered_label)

            if (filtered_label.size > 0):
                new_group = []  # list
                group_NG = []
                group_OK = []
                new_group.append(filtered_label[0])  # choose the first element
                group_NG.append(filtered_label[0])  # choose the first element
                # label[filtered_label[0]] = num_cluster
                filtered_label = np.delete(filtered_label, 0, 0)  # numpy array

                while (len(group_NG) > 0 and filtered_label.size > 0):

                    # the next index will be added into new group
                    picked_index = None
                    # max trong các min
                    max_dis = -1

                    for idx_i in filtered_label:
                        min_dis = 2  # because 1 - cosine_similarity in [0, 2]

                        for idx_j in new_group:
                            # Calculate similarity
                            res = 1 - cosine_similarity(data[idx_i, :].reshape(1, -1), data[idx_j, :].reshape(1, -1))
                            # Choose element having cosine similarity minimum
                            min_dis = res if min_dis > res else min_dis

                        if max_dis < min_dis:
                            max_dis = min_dis
                            picked_index = idx_i

                    new_group.append(picked_index)
                    filtered_label = np.setdiff1d(filtered_label, [picked_index])

                    # print(new_group)
                    isTrue = False

                    for id in group_NG:
                        res2 = 1 - \
                            cosine_similarity(data[id, :].reshape(
                                1, -1), data[picked_index, :].reshape(1, -1))
                        # print(res2)
                        if res2 <= threshold:
                            group_OK.append(id)
                            group_NG.remove(id)
                            isTrue = True

                    if isTrue:
                        group_OK.append(picked_index)
                    else:
                        group_NG.append(picked_index)
                        for id in group_OK:
                            res2 = 1 - \
                                cosine_similarity(data[id, :].reshape(
                                    1, -1), data[picked_index, :].reshape(1, -1))
                            if res2 <= threshold:
                                group_OK.append(picked_index)
                                group_NG.remove(picked_index)
                                break

                    # if filtered_label.size == 0:
                    #     num_cluster = (
                    #         num_cluster - 1) if num_cluster >= 2 else 1
                    #     break
                for id in new_group:
                    label[id] = num_cluster

                num_cluster += 1
            else:
                break
        return label, num_cluster-1


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.epochs = int(self.epochs/2)

    def pack(self, model, loss):
        return {
            "id": self.name,
            "model": model,
            "train_loss": loss,
        }

    # def reply(self, svr_pkg, device):
    #     """
    #     Reply to server with the transmitted package.
    #     The whole local procedure should be planned here.
    #     The standard form consists of three procedure:
    #     unpacking the server_package to obtain the global model,
    #     training the global model, and finally packing the improved
    #     model into client_package.
    #     :param
    #         svr_pkg: the package received from the server
    #     :return:
    #         client_pkg: the package to be send to the server
    #     """
    #     model = self.unpack(svr_pkg)
    #     loss = self.train_loss(model, device)
    #     self.train(model, device)
    #     cpkg = self.pack(model, loss)
    #     return cpkg