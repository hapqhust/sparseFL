from utils import fmodule
from .fedbase import BasicServer, BasicClient
import copy
import math
import wandb
import time
import torch
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        # algorithm hyper-parameters
        self.learning_rate = option['learning_rate']
        self.alpha = option['alpha']
        self.tau = option['tau']
        self.client_last_sample_round = [-1 for i in range(self.num_clients)]
        self.client_grads_history = [0 for i in range(self.num_clients)]
        self.paras_name=['alpha','tau']
        self.selected_clients_r2 = None
        self.phase_one_models = None

    def iterate(self, t):
        # sampling
        self.selected_clients = self.sample()
        # training locally
        self.phase_one_models, _ = self.communicate(self.selected_clients)
        
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
        
        ws, losses = self.communicate(self.selected_clients_r2)
        self.phase_one_models = None
        
        if self.selected_clients == []: return
        
        grads = [self.model - w for w in ws]
        # update GH
        for cid, gi in zip(self.selected_clients, grads):
            self.client_grads_history[cid] = gi
            self.client_last_sample_round[cid] = t

        # project grads
        order_grads = copy.deepcopy(grads)
        order = [_ for _ in range(len(order_grads))]

        # sort client gradients according to their losses in ascending orders
        tmp = sorted(list(zip(losses, order)), key=lambda x: x[0])
        order = [x[1] for x in tmp]

        # keep the original direction for clients with the αm largest losses
        keep_original = []
        if self.alpha > 0:
            keep_original = order[math.ceil((len(order) - 1) * (1 - self.alpha)):]

        # mitigate internal conflicts by iteratively projecting gradients
        for i in range(len(order_grads)):
            if i in keep_original: continue
            for j in order:
                if (j == i):
                    continue
                else:
                    # calculate the dot of gi and gj
                    dot = grads[j].dot(order_grads[i])
                    if dot < 0:
                        order_grads[i] = order_grads[i] - grads[j] * dot / (grads[j].norm()**2)

        # aggregate projected grads
        gt = fmodule._model_average(order_grads)
        # mitigate external conflicts
        if t >= self.tau:
            for k in range(self.tau-1, -1, -1):
                # calculate outside conflicts
                gcs = [self.client_grads_history[cid] for cid in range(self.num_clients) if self.client_last_sample_round[cid] == t - k and gt.dot(self.client_grads_history[cid]) < 0]
                if gcs:
                    g_con = fmodule._model_sum(gcs)
                    dot = gt.dot(g_con)
                    if dot < 0:
                        gt = gt - g_con*dot/(g_con.norm()**2)

        # ||gt||=||1/m*Σgi||
        gnorm = fmodule._model_average(grads).norm()
        gt = gt/gt.norm()*gnorm

        self.model = self.model-gt
        return
    
    def unpack(self, pkgs):
        pkgs = sorted(pkgs, key=lambda x:x['id'])
        ws = [p["model"] for p in pkgs]
        losses = [p["train_loss"] for p in pkgs]
        return ws, losses
    
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
        self.epochs = int(option['num_epochs']/2)

    def reply(self, svr_pkg):
        base_model, model = self.unpack(svr_pkg)
        _, loss = self.test(base_model,'train')
        self.train(model)
        cpkg = self.pack(model, loss)
        return cpkg

    def pack(self, model, loss):

        return {
            "id": self.name,
            "model":model,
            "train_loss":loss,
        }
        
    def unpack(self, received_pkg):
        # unpack the received package
        return received_pkg['base_model'], received_pkg['model']