from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import copy
import torch
import os
import numpy as np
import torch.multiprocessing as mp
import random
import math


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['mu']
        self.phase_one_models = None
        self.phase_one_ids = None
        self.selected_clients_r2 = None

    def iterate(self, t, pool):
        self.selected_clients = sorted(self.sample())
        
        # print(self.selected_clients)
        # Stage 1
        self.phase_one_ids, self.phase_one_models, _ = self.communicate(
            selected_clients = self.selected_clients, pool = pool)
        
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
        self.selected_clients_r2 = shuffled_list
        
        # Stage 2
        _, phase_two_models, _ = self.communicate(selected_clients = self.selected_clients_r2, pool = pool)
        self.phase_one_models = None

        if not self.selected_clients:
            return
        
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in phase_two_models]
        
        self.model = self.aggregate(models, p = [1.0 * (self.client_vols[cid1] + self.client_vols[cid2])/self.data_vol for cid1, cid2 in zip(self.selected_clients, self.selected_clients_r2)])
                                    # p = [1.0 for i in range(len(phase_two_models))])
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

    def pack(self, client_id):
        if not self.phase_one_models:
            send_model = copy.deepcopy(self.model)
        else:
            send_model = self.phase_one_models[self.selected_clients_r2.index(
                client_id)]
            # print(client_id, self.phase_one_ids.index(client_id))
        # print(send_model.cpu().get_device())
        
        model_copy = copy.deepcopy(self.model)
        return {
            "base_model": model_copy.cpu(),
            "model": send_model.cpu(),
        }

    def unpack(self, pkgs):
        pkgs = sorted(pkgs, key=lambda x:x['id'])
        ids = [cp["id"] for cp in pkgs]
        models = [cp["model"] for cp in pkgs]
        train_losses = [cp["train_loss"]
                        for cp in pkgs]
        return ids, models, train_losses

    def communicate(self, selected_clients = None, pool=None):
        packages_received_from_clients = []
        
        if selected_clients:
            packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        # sortlist = sorted(packages_received_from_clients,
        #                   key=lambda d: d['id'])
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, client_id):
        # gpu_id = int(mp.current_process().name[-1]) - 1
        # gpu_id = gpu_id % self.gpus

        # torch.manual_seed(0)
        # torch.cuda.set_device(gpu_id)
        # This is only 'cuda' so its can find the propriate cuda id to train
        device = torch.device(f"cuda:{self.server_gpu_id}")

        # package the necessary information
        svr_pkg = self.pack(client_id)
        # listen for the client's response and return None if the client drops out
        if self.clients[client_id].is_drop():
            return None
        return self.clients[client_id].reply(svr_pkg, device)
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.mu = option['mu']
        self.epochs = int(option['num_epochs']/2)
    
    def reply(self, svr_pkg, device):
        base_model, model = self.unpack(svr_pkg)
        loss = self.train_loss(model, device)
        self.train(base_model, model, device)
        cpkg = self.pack(model, loss)
        return cpkg
    
    def pack(self, model, loss):
        return {
            "id": self.name,
            "model": model.cpu(),
            "train_loss": loss,
        }

    def unpack(self, received_pkg):
        # unpack the received package
        return received_pkg['base_model'], received_pkg['model']

    def train(self, base_model, model, device):
        # global parameters
        model = model.to(device)
        src_model = copy.deepcopy(base_model.to(device))
        src_model.freeze_grad()
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()
                original_loss = self.calculator.get_loss(model, batch_data, device)
                # proximal term
                loss_proximal = 0
                for pm, ps in zip(model.parameters(), src_model.parameters()):
                    loss_proximal += torch.sum(torch.pow(pm-ps,2))
                loss = original_loss + 0.5 * self.mu * loss_proximal                #
                loss.backward()
                optimizer.step()
        return

