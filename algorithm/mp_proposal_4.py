from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import copy
import torch
import numpy as np
import torch.multiprocessing as mp


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['cpg', 'se']
        self.client_per_group = option['cpg']
        self.step_2_epochs = option['se']
        
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
        
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        
        pairings = self.pairing_clients(self.selected_clients, clients_per_group=self.client_per_group)
        # print("Round", t, "pairings: ", pairings)
        
        models = self.communicate(pairings, pool)
        if not self.selected_clients:
            return
        
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        
        self.model = self.aggregate(models, p = [1.0 for i in range(len(models))])
        return
    
    def communicate(self, pairings, pool):
        """
        The whole simulating communication procedure with the pairs of selected clients.
        This part supports for simulating the client dropping out.
        e.g: pairings: [[1,2], [3,4], [5]]
        
        :param
            pairings: the pairings to communicate with
            pool: pool in thread
        :return
            :the unpacked response from pairs that is created ny self.unpack()
        """
        packages_received_from_clients = []        
        packages_received_from_clients = pool.map(self.communicate_with, pairings)
        assert len(packages_received_from_clients) == len(pairings), "Number of models received not equal to number of pairings"
        return packages_received_from_clients

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
        for client_id, epochs in zip(group, local_epochs):
            result_model = self.clients[client_id].reply_peer(result_model, device, epochs)
        return result_model
    
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
    
    def reply_peer(self, model, device, epochs):
        """
        Reply to server with the transmitted package after training with all client in group.   
        """
        self.train(model, device, epochs=epochs)
        return model
    
    def train(self, model, device, epochs=None):
        model = model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        train_epoch = epochs if epochs else self.epochs
        for iter in range(train_epoch):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data, device)
                loss.backward()
                optimizer.step()
        return