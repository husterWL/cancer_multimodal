import torch
import torch.optim as optim
from tqdm import tqdm
from utils.common import roc_draw

class multitrainer():            #训练器

    def __init__(self, config, processor, model, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        
        self.config = config
        self.processor = processor
        self.model = model.to(device)
        self.device = device
     
        parameters = [param for name, param in model.named_parameters()]
        self.optimizer = optim.AdamW(params = parameters, lr = config.learning_rate, weight_decay = config.weight_decay)


    def train(self, train_loader):

        self.model.train()
        loss_list = []
        true_labels, pred_labels = [], []

        for batch in tqdm(train_loader, desc = '----- [Training] '):
            guids, imgs, ehrs, kgs, labels = batch
            imgs, ehrs, kgs, labels = imgs.to(self.device), ehrs.to(self.device), kgs.to(self.device), labels.to(self.device)
            pred, loss = self.model(imgs, ehrs, kgs, labels = labels)

            # guids, imgs, ehrs, labels = batch
            # imgs, ehrs, labels = imgs.to(self.device), ehrs.to(self.device), labels.to(self.device)
            # pred, loss = self.model(imgs, ehrs, labels = labels)
            
            # metric
            loss_list.append(loss.item())
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = round(sum(loss_list) / len(loss_list), 5)
        return train_loss, loss_list  

    def valid(self, val_loader):
        self.model.eval()

        val_loss = 0
        true_labels, pred_labels = [], []

        for batch in tqdm(val_loader, desc = '----- [Validing] '):
            guids, imgs, ehrs, kgs, labels = batch
            imgs, ehrs, kgs, labels = imgs.to(self.device), ehrs.to(self.device), kgs.to(self.device), labels.to(self.device)
            pred, loss = self.model(imgs, ehrs, kgs, labels = labels)

            # guids, imgs, ehrs, labels = batch
            # imgs, ehrs, labels = imgs.to(self.device), ehrs.to(self.device), labels.to(self.device)
            # pred, loss = self.model(imgs, ehrs, labels = labels)

            # metric
            val_loss += loss.item()
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())
            
        metrics, report_dict = self.processor.metric(true_labels, pred_labels)
        return val_loss / len(val_loader), metrics, report_dict
            
    def predict(self, test_loader):

        self.model.eval()   #设置为评估模式
        pred_guids, pred_labels, pred_scores, true_labels = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='----- [Predicting] '):
                guids, imgs, ehrs, kgs, labels = batch
                imgs, ehrs, kgs = imgs.to(self.device), ehrs.to(self.device), kgs.to(self.device)
                pred, scores = self.model(imgs, ehrs, kgs)

                # guids, imgs, ehrs, labels = batch
                # imgs, ehrs = imgs.to(self.device), ehrs.to(self.device)
                # pred, loss = self.model(imgs, ehrs)

                # pred_guids.extend(guids)
                true_labels.extend(labels.tolist())
                pred_labels.extend(pred.tolist())
                pred_scores.extend(scores.tolist())

        # return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]
        metrics, report_dict = self.processor.metric(true_labels, pred_labels)
        roc_draw(true_labels, pred_scores, self.config.output_path + '/roc_curve.jpg')

        return metrics, report_dict