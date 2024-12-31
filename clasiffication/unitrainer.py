import torch
import torch.optim as optim
from tqdm import tqdm

class Trainer():
    def __init__(self, config, processor, model, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.config = config
        self.processor = processor
        self.model = model.to(device)
        self.device = device

        # parameters = list(model.named_parameters())
        parameters = [param for name, param in model.named_parameters()]

        self.optimizer = optim.Adam(params = parameters, lr = config.learning_rate)

    def train(self, train_loader):
        self.model.train()
        loss_list = []
        true_labels, pred_labels = [], []

        for batch in tqdm(train_loader, desc = '-----------【Traing】'):
            # print(type(batch))
            tensors, labels = batch
            # print(type(tensors), type(labels))
            tensors, labels = tensors.to(self.device), labels.to(self.device)
            pred, loss = self.model(tensors, labels=labels)
            loss_list.append(loss.item())
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        train_loss = round(sum(loss_list) / len(loss_list), 5)
        return train_loss, loss_list
    
    def valid(self, valid_lodaer):
        self.model.eval()
        val_loss = 0
        true_labels, pred_labels = [], []

        for batch in tqdm(valid_lodaer, desc='----- [Validing] '):
            tensors, labels = batch
            tensors, labels = tensors.to(self.device), labels.to(self.device)
            pred, loss = self.model(tensors, labels = labels)

            # metric
            val_loss += loss.item()
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())
            
        metrics, report_dict = self.processor.metric(true_labels, pred_labels)
        return val_loss / len(valid_lodaer), metrics, report_dict

    def predict(self, test_loader):
        '''
        重写
        '''
        self.model.eval()
        test_loss = 0
        true_labels, pred_labels = [], []

        for batch in tqdm(test_loader, desc='----- [Predicting] '):
            tensors, labels = batch
            tensors, labels = tensors.to(self.device), labels.to(self.device)
            pred, loss = self.model(tensors, labels = labels)

            test_loss += loss.item()
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())

        # return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]
        metrics, report_dict = self.processor.metric(true_labels, pred_labels)
        return test_loss / len(test_loader), metrics, report_dict