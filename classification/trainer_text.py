import torch
import torch.optim as optim
from tqdm import tqdm
from utils.common import roc_draw
import numpy as np

class trainer_text():            #训练器

    def __init__(self, config, processor, model, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        
        self.config = config
        self.processor = processor
        self.model = model.to(device)
        self.device = device
     
        # parameters = [param for name, param in model.named_parameters()]
        self.optimizer = optim.AdamW([
            {'params': model.clip.parameters(), 'lr': 1e-6}, # CLIP部分用极小学习率
            {'params': model.classifier.parameters(), 'lr': 1e-4} # 分类头用正常学习率
        ])

        self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer = self.optimizer, gamma = config.scheduler_gamma)

    def lr_decay(self):
        self.scheduler.step()

    def train(self, train_loader):

        self.model.train()
        loss_list = []
        true_labels, pred_labels = [], []

        for batch in tqdm(train_loader, desc = '----- [Training] '):
            guids, imgs, ids, masks, kgs, labels = batch
            imgs, ids, masks, labels = imgs.to(self.device), ids.to(self.device), masks.to(self.device), labels.to(self.device)
            pred, loss = self.model(imgs, ids, masks, labels = labels)
            
            # L1 regularization
            # if l1_lambda > 0:
            #     l1_norm = sum(p.abs().sum() for p in self.model.parameters())
            #     loss = loss + l1_lambda * l1_norm

            # metric
            loss_list.append(loss.item())
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

        train_loss = round(sum(loss_list) / len(loss_list), 5)
        return train_loss, loss_list  

    def valid(self, val_loader):
        self.model.eval()

        val_loss = 0
        true_labels, pred_labels = [], []

        for batch in tqdm(val_loader, desc = '----- [Validing] '):
            guids, imgs, ids, masks, kgs, labels = batch
            imgs, ids, masks, labels = imgs.to(self.device), ids.to(self.device), masks.to(self.device), labels.to(self.device)
            pred, loss = self.model(imgs, ids, masks, labels = labels)

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

        for batch in tqdm(test_loader, desc='----- [Predicting] '):
            guids, imgs, ids, masks, kgs, labels = batch
            imgs, ids, masks, labels = imgs.to(self.device), ids.to(self.device), masks.to(self.device), labels.to(self.device)
            pred, scores = self.model(imgs, ids, masks)

            # guids, imgs, ehrs, labels = batch
            # imgs, ehrs = imgs.to(self.device), ehrs.to(self.device)
            # pred, loss = self.model(imgs, ehrs)

            # pred_guids.extend(guids)
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())
            pred_scores.extend(scores.tolist())

        # np.save(self.config.output_path + '/true_labels_multimodal_imgkg_0728_7.npy', np.array(true_labels))
        # np.save(self.config.output_path + '/pred_scores_multimodal_imgkg_0728_7.npy', np.array(pred_scores))

        # return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]
        metrics, report_dict = self.processor.metric(true_labels, pred_labels)
        # roc_draw(true_labels, pred_scores, self.config.output_path + '/' + self.config.model_type + '/roc_curve.png')

        return metrics, report_dict