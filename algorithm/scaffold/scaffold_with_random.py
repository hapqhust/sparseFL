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
    
    def iterate(self, t):
        # sample clients
        self.selected_clients = sorted(self.sample())
        # local training
        dys1, dcs1, models_phase_1 = self.communicate(self.selected_clients)
        
        shuffled_list = self.random(self.selected_clients)
        
        dys2, dcs2, models_phase_2 = self.communicate_phase2(zip(shuffled_list, models_phase_1))
        
        
        dys = [model - self.model for model in models_phase_2] 
        Ks = [math.ceil(self.client_vols[i]/self.option['batch_size']) * 2 for i in shuffled_list]
        dcs = [-1.0 / (K * self.lr) * dy - self.cg for K, dy in zip(Ks, dys)]
        
        
        if self.selected_clients == []: return
        # aggregate
        self.model, self.cg = self.aggregate(dys, dcs)
        return

    def random(self, client_ids):
        new_list = copy.deepcopy(client_ids)
        list_ids = copy.deepcopy(client_ids)
        
        for idx in range(len(new_list)):
            while(True):
                client_id = random.choice(list_ids)
                if (new_list[idx] != client_id):
                    break
            new_list[idx] = client_id
            list_ids.remove[client_id]
        return  new_list

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
