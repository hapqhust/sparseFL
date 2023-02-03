from cmath import isnan
import copy
import torch.nn.functional as F
from torch import nn
import torch
from utils.fmodule import FModule

@torch.no_grad()
def batch_similarity(a, b):
    """
    Args:
        a of shape (x, y)
        b of shape (z, y)
    return:
        c = sim (a, b) of shape (x, z)
    """
    a = torch.softmax(a, dim=1).cpu()
    up = (a @ b.T)
    down = (torch.norm(a, dim=1, keepdim=True) @ torch.norm(b, dim=1, keepdim=True).T)
    val = up / down
    val = torch.nan_to_num(val, 0)
    return val


class MnistMlp(FModule):
    def __init__(self, lowrank=torch.eye(10), bias=False, Phi=None, KDE=None):
        super(MnistMlp, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10, bias=bias)
        
        self.lowrank_mtx = lowrank
        """
        Low rank matrix U, only used for clients, 
        the server uses Phi instead.
        """
        
        self.Phi = Phi
        """
        Phi is the concatenation of all Us (n x n) of all clients (k).
        Phi is a torch tensor of shape k x n x n
            k: the number of clients
            n: the number of classes
        """
        
        self.KDE = KDE
        """
        KDE describes the distribution of samples' representation 
        in each clients.
        KDE is a torch tensor of shape k x d
            k: the number of clients
            d: the dimention of the representation
        """
        
        self.k = self.Phi.shape[0]          # the number of clients
        self.n = self.lowrank_mtx.shape[0]  # the number of classes
        return

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        r_x = F.relu(self.fc2(x))
        return self.masking(r_x)
    
    def masking(self, r_x):
        output = self.fc3(r_x).unsqueeze(2)
        
        mask = None
        if self.training:
            mask = self.lowrank_mtx.unsqueeze(0)
        else:
            b = r_x.shape[0]
            psi_x = batch_similarity(r_x, self.KDE)
            mask = (self.Phi.view(self.k, -1).T @ psi_x.unsqueeze(2)).view(b, self.n, self.n)
        
        return (mask.to(self.device) @ output).squeeze(2)
    
    def get_representation(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        r_x = F.relu(self.fc2(x))
        return r_x
    

class MnistCnn(FModule):
    def __init__(self, lowrank=torch.eye(10), bias=False, Phi=None, KDE=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 10, bias=bias)

        self.lowrank_mtx = lowrank.to(dtype=torch.float32)
        """
        Low rank matrix U, only used for clients, 
        the server uses Phi instead.
        """
        
        self.Phi = Phi.to(dtype=torch.float32)
        """
        Phi is the concatenation of all Us (n x n) of all clients (k).
        Phi is a torch tensor of shape k x n x n
            k: the number of clients
            n: the number of classes
        """
        
        self.KDE = KDE.to(dtype=torch.float32)
        """
        KDE describes the distribution of samples' representation 
        in each clients.
        KDE is a torch tensor of shape k x d
            k: the number of clients
            d: the dimention of the representation
        """
        
        self.k = self.Phi.shape[0]          # the number of clients
        self.n = self.lowrank_mtx.shape[0]  # the number of classes
        
        self.apply_masking = False
        return
    
    def forward(self, x):
        r_x = self.encoder(x)
        return self.masking(r_x)

    def encoder(self, x):
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        return x

    def decoder(self, x):
        x = self.fc2(x)
        return x
    
    def masking(self, r_x):
        output = self.decoder(r_x).unsqueeze(2)
        mask = None
        if self.training:
            mask = (self.lowrank_mtx @ self.lowrank_mtx).unsqueeze(0)
        else:
            b = r_x.shape[0]
            psi_x = batch_similarity(r_x, self.KDE) 
            mask = (self.Phi.view(self.k, -1).T @ psi_x.unsqueeze(2)).view(b, self.n, self.n)
        return (mask.to("cuda" if output.is_cuda else "cpu").to(torch.float32) @ output).squeeze(2)
        
    def pred_and_rep(self, x):
        e = self.encoder(x)
        o = self.masking(e)
        return o, e
    
    def update_KDE(self, index, KDE_i):
        self.KDE[index] = copy.copy(KDE_i)
        return 
        
    def update_mask(self, mask):
        self.lowrank_mtx = mask
        return