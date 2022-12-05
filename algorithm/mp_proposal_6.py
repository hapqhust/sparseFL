import copy
from .mp_fedbase import MPBasicServer, MPBasicClient
from algorithm.utils.smt_utils.smt_utils import manual_classifier_aggregate, generate_Us
from algorithm.utils.smt_utils.mnist import MnistCnn
from algorithm.utils.alg_utils.alg_utils import KDR_loss
from utils import fmodule
from utils.fmodule import get_module_from_model
import torch

def rewrite_classifier(base_model, model):
    base_classifier = get_module_from_model(base_model)[-1]._parameters['weight']
    model_classifier = get_module_from_model(model)[-1]._parameters['weight']
    
    base_classifier.copy_(model_classifier)
    return

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        
        self.U_list = generate_Us(k=self.num_clients, dim=10)
        Phi = torch.stack(self.U_list, dim=0)
        self.model = MnistCnn(Phi=Phi, KDE=torch.zeros(self.num_clients, 512))
        self.warm_ups = 100

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses, KDEs, indexes = self.communicate(self.selected_clients, pool)
        if not self.selected_clients: 
            return
        
        if t > self.warm_ups:
            for index, KDEi in zip(indexes, KDEs):
                self.model.update_KDE(index, KDEi)
            
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        
        impact_factors = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]
        self.model = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, impact_factors)])
        
        if t > self.warm_ups:
            head = fmodule._model_sum([model_k for model_k in models])
            rewrite_classifier(self.model, head)
        return

    def unpack(self, packages_received_from_clients):
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        KDEs = [cp["KDE"] for cp in packages_received_from_clients]
        indexes = [cp["id"] for cp in packages_received_from_clients]
        return models, train_losses, KDEs, indexes
    
    def pack(self, client_id):
        mask = self.U_list[client_id]
        send_model = copy.deepcopy(self.model)
        send_model.update_mask(mask)
        
        return {
            "model" : send_model,
        }
    
class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = torch.nn.CrossEntropyLoss()
    
    def reply(self, svr_pkg, device):
        model= self.unpack(svr_pkg)
        loss = self.train_loss(model, device)
        KDE = self.train(model, device)
        cpkg = self.pack(model, loss, KDE)
        return cpkg
    
    def pack(self, model, loss, KDE):
        return {
            "id" : self.name,
            "model" : model,
            "train_loss": loss,
            "KDE": KDE
        }
        
    def train(self, model, device):
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        repres_list = []
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss, representations = self.get_loss(model, src_model, batch_data, device)
                loss.backward()
                optimizer.step()
                
                if iter == self.epochs - 1:
                    repres_list.append(representations.detach().cpu())
        
        repres = torch.cat(repres_list, dim=0)
        KDE = self.process_reps(repres)
        return KDE
    
    def get_loss(self, model, src_model, data, device):
        tdata = self.calculator.data_to_device(data, device)    
        output_s, representation_s = model.pred_and_rep(tdata[0])                  # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])                    # Teacher
        kl_loss = KDR_loss(representation_t, representation_s, device)        # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        # print("representations")
        # print(representation_s)
        # print("outputs")
        # print(output_s)
        # exit(0)
        return loss + kl_loss, representation_s
    
    def process_reps(self, representations):
        return torch.sum(torch.softmax(representations, dim=1), dim=0, keepdim=True) / representations.shape[0]
        