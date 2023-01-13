import math
import random
from ..fedbase import BasicServer, BasicClient
import copy
from utils import fmodule
import time, wandb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.eta = option['eta']
        self.cg = self.model.zeros_like()
        self.paras_name = ['eta']

    def pack(self, client_id):
        return {
            "model": copy.deepcopy(self.model),
            "cg": self.cg,
        }

    def unpack(self, pkgs):
        pkgs = sorted(pkgs, key=lambda x:x['id'])
        dys = [p["dy"] for p in pkgs]
        dcs = [p["dc"] for p in pkgs]
        models = [p["model"] for p in pkgs]
        return dys, dcs, models
    
    def clustering(self, client_ids, dys1):
        clt = [dy[-1].detach().numpy() for dy in dys1]

        # clt = torch.FloatTensor(np.array(clt))
        data = np.asarray(clt, dtype=float)
        data = self.unit_scaler(data)
        N = len(data)
        label, num_clusters = self.classifier(data, N, threshold=1.25)
        return 

    def iterate(self, t):
        # sample clients
        self.selected_clients = sorted(self.sample())
        # local training
        dys1, dcs1, models_phase_1 = self.communicate(self.selected_clients)
        
        shuffled_list = self.clustering(self.selected_clients, dys1)
        
        dys2, dcs2, models_phase_2 = self.communicate_phase2(zip(shuffled_list, models_phase_1))
        
        
        dys = [model - self.model for model in models_phase_2] 
        Ks = [math.ceil(self.client_vols[i]/self.option['batch_size']) * 2 for i in shuffled_list]
        dcs = [-1.0 / (K * self.lr) * dy - self.cg for K, dy in zip(Ks, dys)]
        
        
        if self.selected_clients == []: return
        # aggregate
        self.model, self.cg = self.aggregate(dys, dcs)
        return
    
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
                            print(res2)
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

    def aggregate(self, dys, dcs):  # c_list is c_i^+
        dw = fmodule._model_average(dys)
        dc = fmodule._model_average(dcs)
        new_model = self.model + self.eta * dw
        new_c = self.cg + 1.0 * len(dcs) / self.num_clients * dc
        return new_model, new_c


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.c = None
        
    def train(self, model, cg):
        model.train()
        if not self.c:
            self.c = model.zeros_like()
            self.c.freeze_grad()
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        cg.freeze_grad()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        num_batches = 0
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data)
                loss.backward()
                for pm, pcg, pc in zip(model.parameters(), cg.parameters(), self.c.parameters()):
                    pm.grad = pm.grad - pc + pcg
                optimizer.step()
                num_batches += 1
        # update local control variate c
        K = num_batches
        dy = model - src_model
        dc = -1.0 / (K * self.learning_rate) * dy - cg
        self.c = self.c + dc
        return dy, dc

    def reply(self, svr_pkg):
        model, c_g = self.unpack(svr_pkg)
        dy, dc = self.train(model, c_g)
        cpkg = self.pack(dy, dc, model)
        return cpkg

    def pack(self, dy, dc, model):
        return {
            "id": self.name,
            "dy": dy,
            "dc": dc,
            "model": model
        }

    def unpack(self, received_pkg):
        # unpack the received package
        return received_pkg['model'], received_pkg['cg']
