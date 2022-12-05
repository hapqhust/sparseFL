from .mp_fedbase import MPBasicServer, MPBasicClient
from .utils.alg_utils.alg_utils import get_ultimate_layer, KDR_loss, rewrite_classifier
import torch
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, features, outputs):
        self.data = features
        self.pred = outputs
        assert self.data.shape[0] == self.pred.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        label = self.pred[item]
        return data, label
    
    
class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.clusters = [[0,1,2,3,4], [5,6,7,8,9]]
        # self.paras_name = ['distill_method']
    
    def select_cluster_head(self, clusters, time_step):
        """
        This method choose a head out of a cluster.
        return:
            [
                {
                    "head": 0,
                    "members": 1,2,3,4
                },
                {
                    "head": 5,
                    "members": 6,7,8,9
                }
            ]
        """
        heads = []
        for cluster in clusters:
            head = cluster[time_step % len(cluster)]
            members = cluster.copy()
            members.remove(head)
            heads.append({"head": head, "members": members})
            
        return heads
    
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        cluster_dict = self.select_cluster_head(self.clusters, t)
        heads = [cluster["head"] for cluster in cluster_dict]
        
        # Phase one communication
        phase_one_participants = list(set(self.selected_clients) - set(heads))
        phase_one_models, train_losses, metadata = self.communicate(phase_one_participants, pool)
        self.update_clustering()
        
        # Phase two communication
        phase_two_models = self.communicate_heads(cluster_dict, metadata, phase_one_participants, pool)
        
        if not self.selected_clients:
            return

        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        phase_two_models = [model.to(device0) for model in phase_two_models]
        
        impact_factors = [1.0/len(phase_two_models) for i in range(len(phase_two_models))]
        self.model = self.aggregate(phase_two_models, p=impact_factors)
        
        torch.save(self.model.cpu().state_dict(), f"./exp/assemble_round_{t}.pt")
        return
    
    def communicate(self, selected_clients, pool):
        packages_received_from_clients = []        
        packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
        self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        sortlist = sorted(packages_received_from_clients, key=lambda d: d['id'])
        return self.unpack(sortlist)
    
    def unpack(self, packages_received_from_clients):
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        metadata = [cp["metadata"] for cp in packages_received_from_clients]
        return models, train_losses, metadata
    
    def update_clustering(self):
        pass
    
    def communicate_heads(self, cluster_dict, metadata, indexes, pool):
        """
        cluster_dict = [{"head": 0, "members": [1,2,3,4]}, {"head": 5, "members": [6,7,8,9]}]
        metadata = [meta_1, meta_2, meta_3, meta_4, meta_6, meta_7, meta_8, meta_9]
        indexes = [1, 2, 3, 4, 6, 7, 8, 9]
        """
        comm_dict = []
        for cluster in cluster_dict:
            member_meta = []
            for member_idx in cluster['members']:
                member_meta.append(metadata[indexes.index(member_idx)])
            
            comm_dict.append(
                {
                    "head": cluster["head"],
                    "metadata": member_meta
                }
            )
        
        packages_received_from_clients = []
        packages_received_from_clients = pool.map(self.communicate_with_heads, comm_dict)
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        return packages_received_from_clients
    
    def communicate_with_heads(self, comm_infor):
        gpu_id = int(mp.current_process().name[-1]) - 1
        gpu_id = gpu_id % self.gpus

        torch.manual_seed(0)
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda')

        head_id = comm_infor["head"]
        metadata = comm_infor["metadata"]
        
        if self.clients[head_id].is_drop(): 
            return None
        
        return self.clients[head_id].phase_two_training(self.model.cpu(), metadata, device)
    
    
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.metadata = None
        
    def pack(self, model, loss):
        return {
            "id" : self.name,
            "model" : model,
            "train_loss": loss,
            "metadata": self.metadata,
        }

    def train(self, model, device):
        model = model.to(device)
        model.train()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        self.metadata = {"imm_features": None, "outputs": None}
        
        for epoch in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.get_loss(model, batch_data, device, last_epoch=True if (epoch == self.epochs - 1) else False)
                loss.backward()
                optimizer.step()
        return
    
    def get_loss(self, model, data, device, last_epoch):
        tdata = self.calculator.data_to_device(data, device)
        output, imm = model.pred_and_imm(tdata[0])
        loss = self.lossfunc(output, tdata[1])
        
        if last_epoch:
            imm = imm.detach().cpu()
            output = output.detach().cpu()
            
            if self.metadata["imm_features"] is not None:
                self.metadata["imm_features"] = torch.cat([self.metadata["imm_features"], imm], dim=0)
                self.metadata["outputs"] = torch.cat([self.metadata["outputs"], output], dim=0)
            else:
                self.metadata["imm_features"] = imm
                self.metadata["outputs"] = output
            
        return loss
    
    def phase_two_training(self, model, metadata, device):
        model = model.to(device)
        model.train()
        torch.save(model.state_dict(), f"./exp/head_{self.name}_before_training.pt")
        # metadata processing
        abstract_features = torch.cat([client_meta["imm_features"] for client_meta in metadata], dim=0)
        outputs = torch.cat([client_meta["outputs"] for client_meta in metadata], dim=0)
        
        meta_dataset = MyDataset(abstract_features, outputs)
        
        meta_loader = self.calculator.get_data_loader(meta_dataset, batch_size=self.batch_size)
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data, device)
                loss.backward()
                optimizer.step()
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(meta_loader):
                model.zero_grad()
                loss = self.get_meta_loss(model, batch_data, device)
                loss.backward()
                optimizer.step()
                
        torch.save(model.cpu().state_dict(), f"./exp/head_{self.name}_after_training.pt")
        return model
    
    def get_meta_loss(self, model, batch_meta, device):
        tdata = self.calculator.data_to_device(batch_meta, device)
        output = model.forward_imm(tdata[0])
        loss = self.lossfunc(output, tdata[1])
        return loss