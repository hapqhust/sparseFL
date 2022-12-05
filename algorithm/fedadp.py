from .fedbase import BasicServer, BasicClient
import copy
import torch
import wandb
import time

def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def flatten_model(model):
    ten = torch.cat([flatten_tensors(i) for i in model.parameters()])
    return ten


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.smooth_angle = None
        
    def get_impact_factor(self, models, round):
        """
        :param local_weights the weights of model after SGD updates
        :param global_weight the weight of the global model
        """
        flat_models = [flatten_model(model) for model in models]
        lodal_weights = torch.vstack(flat_models)
        
        model_difference = lodal_weights.to('cpu') - flatten_model(self.model).to('cpu')

        F_i = - model_difference / 0.01

        D_i = torch.tensor([self.client_vols[cid] for cid in self.selected_clients])
        D_i = torch.FloatTensor(D_i / torch.sum(D_i)).to('cpu')

        F = D_i.transpose(-1,0) @ F_i

        corel = F.unsqueeze(0) @ F_i.transpose(-1,0)

        instantaneous_angle = torch.squeeze(torch.arccos(corel/(torch.norm(F_i) * torch.norm(F)))).to('cpu')

        if (self.smooth_angle is None):
            self.smooth_angle = instantaneous_angle
        else:
            self.smooth_angle = (round - 1)/round * self.smooth_angle + 1/round * instantaneous_angle

        impact_factor = torch.squeeze(5 * (1 - torch.exp( - torch.exp(- 5 * (self.smooth_angle - torch.ones_like(self.smooth_angle))))))        
        return impact_factor.cpu().tolist()
    
    
    def iterate(self, t):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients)
        if not self.selected_clients: 
            return
        
        start = time.time()
        impact_factor = self.get_impact_factor(models, t)
        self.model = self.aggregate(models, p = impact_factor)
        end = time.time()
        if self.wandb:
            wandb.log({"Aggregation_time": end-start})
        return


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
