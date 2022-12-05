from itertools import chain
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
import copy

class Feature_generator(FModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3136, 512)

    def forward(self, x):
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        return x

class Classifier(FModule):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.softmax(self.fc2(x), dim=0)
        return x
    
    def flatten(self):
        return copy.deepcopy(self.fc2.weight).detach().cpu()


class MyModel(FModule):
    def __init__(self):
        super().__init__()
        self.feature_generator = Feature_generator()
        self.classifier = Classifier()
        
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        x = self.feature_generator(x)
        x = self.classifier(x)
        return x
    
    def pred_and_rep(self, x):
        e = self.feature_generator(x)
        o = self.classifier(e)
        return o, e
    
    def eval(self):
        self.feature_generator.eval()
        self.classifier.eval()
        return
        
    def train(self, mode=True):
        self.feature_generator.train(mode)
        self.classifier.train(mode)
        return
        
    def freeze_grad(self):
        self.feature_generator.freeze_grad()
        self.classifier.freeze_grad()
        return self
    
    def zero_grad(self):
        self.feature_generator.zero_grad()
        self.classifier.zero_grad()
        return
        
    def to(self, option):
        self.feature_generator = self.feature_generator.to(option)
        self.classifier = self.classifier.to(option)
        return self
        
    def update(self, newFgenerator, newClassifier):
        self.feature_generator = newFgenerator
        self.classifier = newClassifier
        return
    
    def parameters(self):
        return chain(self.feature_generator.parameters(), self.classifier.parameters())        