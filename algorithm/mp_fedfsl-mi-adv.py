from .mp_fedbase import MPBasicServer, MPBasicClient
import torch, copy
from itertools import chain
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
from benchmark.toolkits import ClassifyCalculator

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

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

class MyCalculator(ClassifyCalculator):
    @torch.no_grad()
    def test(self, feature_generator, classifier, data, device=None):
        """Metric = Accuracy"""
        tdata = self.data_to_device(data, device)
        feature_generator = feature_generator.to(device)
        classifier = classifier.to(device)

        feature = feature_generator(tdata[0])
        outputs = classifier(feature)
        loss = self.lossfunc(outputs, tdata[-1])

        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item(), loss.item()


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.feature_generator = Feature_generator()
        self.classifier = Classifier()
        self.calculator = MyCalculator(device=f'cuda:{self.server_gpu_id}')
        self.device = torch.device(f'cuda:{self.server_gpu_id}')
        return

    def pack(self, client_id):
        return {
            "fgenerator" : copy.deepcopy(self.feature_generator),
            "classifier" : copy.deepcopy(self.classifier)
        }

    def unpack(self, packages_received_from_clients):
        fgenerators = [cp["fgenerator"] for cp in packages_received_from_clients]
        classifiers = [cp["classifier"] for cp in packages_received_from_clients]
        return fgenerators, classifiers

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        fgenerators, classifiers = self.communicate(self.selected_clients, pool)
        
        fgenerators = [model.to(self.device) for model in fgenerators]
        classifiers = [model.to(self.device) for model in classifiers]
        
        if not self.selected_clients: 
            return

        impact_factors = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]
        self.feature_generator = self.aggregate(fgenerators, p = impact_factors)
        self.classifier = self.aggregate(classifiers, p = impact_factors)
        return

    def test(self, model=None, device='cuda'):
        if self.test_data:
            self.feature_generator.eval()
            self.classifier.eval()
            loss = 0
            eval_metric = 0
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                bmean_eval_metric, bmean_loss = self.calculator.test(self.feature_generator, self.classifier, batch_data, device)                
                loss += bmean_loss * len(batch_data[1])
                eval_metric += bmean_eval_metric * len(batch_data[1])
            eval_metric /= len(self.test_data)
            loss /= len(self.test_data)
            return eval_metric, loss
        else: 
            return -1,-1
    
    def test_on_clients(self, round, dataflag='valid', device='cuda'):
        evals, losses = [], []
        for c in self.clients:
            eval_value, loss = c.test(self.feature_generator, self.classifier, dataflag, device)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.epochs = int(self.epochs/2)
        self.divergence = nn.KLDivLoss()
        self.lossfunc = nn.CrossEntropyLoss()
        self.calculator = MyCalculator('cuda')

    def phase_one_training(self, fgenerator, classifier, Adv_classifier, device):
        freeze_model(fgenerator)
        unfreeze_model(classifier)
        unfreeze_model(Adv_classifier)
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = torch.optim.Adam(chain(classifier.parameters(), Adv_classifier.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay, amsgrad=True)

        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                fgenerator.zero_grad()
                classifier.zero_grad()

                tdata = self.calculator.data_to_device(batch_data, device)
                feature = fgenerator(tdata[0])
                outputs = classifier(feature)
                adv_outputs = Adv_classifier(feature)

                loss = self.lossfunc(outputs, tdata[1])
                adv_loss = self.lossfunc(adv_outputs, tdata[1])
                adv_divergence = self.divergence(outputs, adv_outputs)
                
                total_loss = loss + adv_loss - 0.1 * adv_divergence
                
                total_loss.backward()
                optimizer.step()
        return

    def phase_two_training(self, fgenerator, classifier, Adv_classifier, device):
        unfreeze_model(fgenerator)
        freeze_model(classifier)
        freeze_model(Adv_classifier)
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = torch.optim.Adam(fgenerator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, amsgrad=True)

        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                fgenerator.zero_grad()
                classifier.zero_grad()

                tdata = self.calculator.data_to_device(batch_data, device)
                feature = fgenerator(tdata[0])
                outputs = classifier(feature)
                adv_outputs = Adv_classifier(feature)

                loss = self.lossfunc(outputs, tdata[1])
                adv_loss = self.lossfunc(adv_outputs, tdata[1])
                adv_divergence = self.divergence(outputs, adv_outputs)

                total_loss = loss + adv_loss + 0.1 * adv_divergence
                total_loss.backward()
                optimizer.step()
        return

    def unpack(self, received_pkg):
        return received_pkg['fgenerator'], received_pkg['classifier']

    def reply(self, svr_pkg, device):
        fgenerator, classifier = self.unpack(svr_pkg)
        Adv_classifier = copy.deepcopy(classifier)
        # Adv_classifier = Classifier()

        fgenerator = fgenerator.to(device)
        classifier = classifier.to(device)
        Adv_classifier = Adv_classifier.to(device)

        self.phase_one_training(fgenerator, classifier, Adv_classifier, device)
        self.phase_two_training(fgenerator, classifier, Adv_classifier, device)

        cpkg = self.pack(fgenerator, classifier)
        return cpkg

    def pack(self, fgenerator, classifier):
        return {
            "fgenerator" : fgenerator,
            "classifier" : classifier,
        }

    def test(self, feature_generator, classifier, dataflag='valid', device='cuda'):
        dataset = self.train_data
        feature_generator.eval()
        classifier.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=4)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss = self.calculator.test(feature_generator, classifier, batch_data, device)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss