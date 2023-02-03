from utils import fmodule
from .fedbase import BasicServer, BasicClient
import numpy as np
import copy
import torch
import time, wandb

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        # self.m = fmodule._modeldict_zeroslike(self.model.state_dict())
        self.m = copy.deepcopy(self.model) * 0.0
        self.beta = option['beta']
        self.alpha = 1.0 - self.beta
        self.gamma = option['gamma']
        self.eta = option['learning_rate']
        self.paras_name=['beta','gamma']
        self.selected_clients_r2 = None
        self.phase_one_models = None

    def unpack(self, pkgs):
        pkgs = sorted(pkgs, key=lambda x:x['id'])
        ws = [p["model"] for p in pkgs]
        losses = [p["train_loss"] for p in pkgs]
        ACC = [p["acc"] for p in pkgs]
        freq = [p["freq"] for p in pkgs]
        return ws, losses, ACC, freq
    
    def pack(self, client_id):
        if not self.phase_one_models:
            send_model = copy.deepcopy(self.model)
        else:
            send_model = self.phase_one_models[self.selected_clients_r2.index(
                client_id)]
        
        model_copy = copy.deepcopy(self.model)
        return {
            "base_model": model_copy.cpu(),
            "model": send_model.cpu(),
        }

    def iterate(self, t):
        # sample clients
        self.selected_clients = sorted(self.sample())
        print(self.selected_clients)

        # training
        self.phase_one_models, losses, _, _ = self.communicate(self.selected_clients)
        
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
        
        ws, losses, ACC, F = self.communicate(self.selected_clients_r2)
        self.phase_one_models = None
        
        if self.selected_clients == []: return
        
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        ws = [i.to(device0) for i in ws]
        
        # aggregate
        # calculate ACCi_inf, fi_inf
        sum_acc = np.sum(ACC)
        sum_f = np.sum(F)
        ACCinf = [-np.log2(1.0*acc/sum_acc+0.000001) for acc in ACC]
        Finf = [-np.log2(1-1.0*f/sum_f+0.00001) for f in F]
        sum_acc = np.sum(ACCinf)
        sum_f = np.sum(Finf)
        ACCinf = [acc/sum_acc for acc in ACCinf]
        Finf = [f/sum_f for f in Finf]
        # calculate weight = αACCi_inf+βfi_inf
        p = [self.alpha*accinf+self.beta*finf for accinf,finf in zip(ACCinf,Finf)]
        wnew = self.aggregate(ws, p)
        dw = wnew - self.model
        # calculate m = γm+(1-γ)dw
        self.m = self.gamma * self.m + (1 - self.gamma) * dw
        self.model = wnew - self.m * self.eta
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
    
    def communicate_with(self, client_id):
        # package the necessary information
        svr_pkg = self.pack(client_id)
        # listen for the client's response and return None if the client drops out
        if self.clients[client_id].is_drop(): return None
        return self.clients[client_id].reply(svr_pkg)

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.frequency = 0
        self.momentum = option['gamma']
        self.epochs = int(option['num_epochs']/2)

    def reply(self, svr_pkg):
        base_model, model = self.unpack(svr_pkg)
        acc, loss = self.test(base_model,'train')
        self.train(model)
        cpkg = self.pack(model, loss, acc)
        return cpkg

    def pack(self, model, loss, acc):
        self.frequency += 0.5

        return {
            "id": self.name,
            "model":model,
            "train_loss":loss,
            "acc":acc,
            "freq":self.frequency,
        }
        
    def unpack(self, received_pkg):
        # unpack the received package
        return received_pkg['base_model'], received_pkg['model']
        