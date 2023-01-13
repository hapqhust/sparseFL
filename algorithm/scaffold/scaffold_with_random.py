import math
from multiprocessing.pool import ThreadPool
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
    
    def iterate(self, t):
        # sample clients
        self.selected_clients = sorted(self.sample())
        print(self.selected_clients)
        # local training
        dys1, dcs1, models_phase_1 = self.communicate(self.selected_clients)
        
        pairings = self.pairing_clients(self.selected_clients)
        shuffled_list = copy.deepcopy(self.selected_clients)
        print(pairings)
        for pair in pairings:
            if(len(pair) == 2):
                idx1 = self.selected_clients.index(pair[0])
                idx2 = self.selected_clients.index(pair[1])
                shuffled_list[idx1] = pair[1]
                shuffled_list[idx2] = pair[0]
        print(shuffled_list)
        
        dys2, dcs2, models_phase_2 = self.communicate_phase2(zip(shuffled_list, models_phase_1))
        
        
        dys = [model - self.model for model in models_phase_2] 
        Ks = [math.ceil(self.client_vols[i]/self.option['batch_size']) * 2 for i in shuffled_list]
        dcs = [-1.0 / (K * self.lr) * dy - self.cg for K, dy in zip(Ks, dys)]
        
        
        if self.selected_clients == []: return
        # aggregate
        self.model, self.cg = self.aggregate(dys, dcs)
        return

    def pairing_clients(self, clients, clients_per_group=2):
        """
        clients = [0,1,2,3,4,5,6,....] the list of client's id
        """
        participants = clients.copy()
        pairs = []
        
        while len(participants) > 1:
            one_pair = list(np.random.choice(participants, clients_per_group, replace=False))
            pairs.append(one_pair)
            participants = list(set(participants) - set(one_pair))
        
        if len(participants):
            pairs.append(participants)
        
        return pairs
    
    # def random(self, client_ids):
    #     new_list = copy.deepcopy(client_ids)
    #     list_ids = copy.deepcopy(client_ids)
        
    #     for idx in range(len(new_list)):
    #         while(True):
    #             client_id = random.choice(list_ids)
    #             if (new_list[idx] != client_id):
    #                 print(list_ids)
    #                 break
    #         new_list[idx] = client_id
    #         list_ids.remove(client_id)
    #     return  new_list
    
    def communicate_phase2(self, groups):
        packages_received_from_clients = []
        if self.num_threads <= 1:
            # computing iteratively
            for group in groups:
                response_from_client_id = self.communicate_with_phase2(group)
                packages_received_from_clients.append(response_from_client_id)
        else:
            # computing in parallel
            pool = ThreadPool(min(self.num_threads, len(groups)))
            packages_received_from_clients = pool.map(self.communicate_with_phase2, groups)
            pool.close()
            pool.join()
        # count the clients not dropping
        # self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        return self.unpack(packages_received_from_clients)
    
    def communicate_with_phase2(self, group):
        client_id, model = group
        # package the necessary information
        svr_pkg = {
            "model": copy.deepcopy(model),
            "cg": self.cg,
        }
        # listen for the client's response and return None if the client drops out
        if self.clients[client_id].is_drop(): return None
        return self.clients[client_id].reply(svr_pkg)

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
