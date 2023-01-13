from pathlib import Path

from sparseFL.algorithm.mp_fedbase import MPBasicClient, MPBasicServer

import copy
import torch
import numpy as np
import torch.multiprocessing as mp
from utils import fmodule
import time, wandb


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['cpg', 'se']
        self.client_per_group = option['cpg']
        self.step_2_epochs = int(option['num_epochs']/2)
        
        #scaffold
        self.eta = option['eta']
        self.cg = self.model.zeros_like()
        self.paras_name = ['eta']
        
    def iterate(self, t, pool):
        # sample clients
        self.selected_clients = self.sample()
        
        pairings = self.pairing_clients(self.selected_clients, clients_per_group=self.client_per_group)
        # print("Round", t, "pairings: ", pairings)
        
        dys, dcs = self.communicate(pairings, pool)
        if not self.selected_clients:
            return
        
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
    
    def unpack(self, pkgs):
        dys = [p["dy"] for p in pkgs]
        dcs = [p["dc"] for p in pkgs]
        return dys, dcs
    
    def communicate(self, pairings, pool):
        packages_received_from_clients = []        
        packages_received_from_clients = pool.map(self.communicate_with, pairings)
        assert len(packages_received_from_clients) == len(pairings), "Number of models received not equal to number of pairings"
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, group):
        """
        e.g: group = [1,2]
        :param
            group: the group of the clients to communicate with
        :return
            client_package: the reply from the client and will be 'None' if losing connection

        """
        gpu_id = int(mp.current_process().name[-1]) - 1
        gpu_id = gpu_id % self.gpus

        torch.manual_seed(0)
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda')
        
        local_epochs = [self.step_2_epochs for i in group]  # [2,2,2,...]
        # local_epochs[0] = None                              # [None,2,2,...]
        
        result_model = copy.deepcopy(self.model)
        old_model = copy.deepcopy(self.model) 
        new_cg = self.cg
        num_batches = 0
        for client_id, epochs in zip(group, local_epochs):
            result_model, new_cg, num_batches, dy, dc = self.clients[client_id].reply_peer(result_model, old_model, new_cg, num_batches, device, epochs)
        return {
            "dy": dy,
            "dc": dc,
        }

    def aggregate(self, dys, dcs):  # c_list is c_i^+
        dw = fmodule._model_average(dys)
        dc = fmodule._model_average(dcs)
        new_model = self.model + self.eta * dw
        new_c = self.cg + 1.0 * len(dcs) / self.num_clients * dc
        return new_model, new_c        
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.c = None

    
    def reply_peer(self, model, old_model, cg, num_batches, device, epochs):
        """
        Reply to server with the transmitted package after training with all client in group.   
        """
        return self.train(model, old_model, cg, num_batches, device, epochs=epochs)
    
    def train(self, model, old_model, cg, num_batches, device, epochs=None):
        model = model.to(device)
        model.train()
        if not self.c:
            self.c = model.zeros_like()
            self.c.freeze_grad()
            
        # global parameters
        src_model = old_model
        src_model.freeze_grad()
        cg.freeze_grad()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        train_epoch = epochs if epochs else self.epochs
        for iter in range(train_epoch):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data, device)
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
        return model, cg, num_batches, dy, dc