import torch
from utils.fmodule import get_module_from_model
import copy

@torch.no_grad()
def manual_classifier_aggregate(final_model, models, p):
    final_classifier = copy.deepcopy(get_module_from_model(models[0])[-1]._parameters['weight'] / p[0])
    for model, pk in zip(models[1:], p):
        final_classifier.add_(get_module_from_model(model)[-1]._parameters['weight'] / pk)
    
    get_module_from_model(final_model)[-1]._parameters['weight'].copy_(final_classifier)
    return final_model

def generate_Us(k, dim):
    G = torch.rand([dim, dim], dtype=torch.float64)
    H = torch.linalg.inv(G)
    U_list = []
    for i in range(k):
        R = torch.zeros_like(H)
        R[i,i] = 1
        U_i = G @ R @ H
        
        U_list.append(U_i)
    return U_list

