import torch
import torch.optim as optim
from tqdm import tqdm
from utils.common import roc_draw


class Trainer():
    def __init__(self, config, processor, model, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.config = config
        self.processor = processor
        self.model = model.to(device)
        self.device = device

        # parameters = list(model.named_parameters())
        parameters = [param for name, param in model.named_parameters()]

        self.optimizer = optim.Adam(params = parameters, lr = config.learning_rate, weight_decay = config.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer = self.optimizer, gamma = config.scheduler_gamma)

    def lr_decay(self):
        self.scheduler.step()

    def train(self, train_loader):
        self.model.train()
        loss_list = []
        true_labels, pred_labels = [], []

        for batch in tqdm(train_loader, desc = '-----------[Training]'):
            # print(type(batch))
            guids, tensors, labels = batch
            # print(type(tensors), type(labels))
            tensors, labels = tensors.to(self.device), labels.to(self.device)
            pred, loss = self.model(tensors, labels = labels)

            '''
            测试用
            '''
            # features, pred, loss = self.model(tensors, labels = labels)
            # asset_dict = {
            #     'features': features.detach().cpu().numpy().astype(np.float32),
            #     'guids': guids,
            # }
            # print(asset_dict)
            # with open(self.config.output_path + '/features.json', 'w') as f:
            #     json.dump(asset_dict, f, indent = 4)
            
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
            guids, tensors, labels = batch
            tensors, labels = tensors.to(self.device), labels.to(self.device)
            pred, loss = self.model(tensors, labels = labels)

            # asset_dict = {
            #     'features': features.detach().cpu().numpy().astype(np.float32),
            #     'guids': guids,
            # }
            # print(asset_dict)

            # metric
            val_loss += loss.item()
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())
            
        metrics, report_dict = self.processor.metric(true_labels, pred_labels)
        return val_loss / len(valid_lodaer), metrics, report_dict

    def predict(self, test_loader):

        self.model.eval()
        test_loss = 0
        true_labels, pred_labels, pred_scores = [], [], []

        for batch in tqdm(test_loader, desc='----- [Predicting] '):
            guids, tensors, labels = batch
            tensors, labels = tensors.to(self.device), labels.to(self.device)
            pred, scores = self.model(tensors)


            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())
            pred_scores.extend(scores.tolist())

        # np.save(self.config.output_path + '/true_labels_unimodal_univision_0723_3.npy', np.array(true_labels))
        # np.save(self.config.output_path + '/pred_scores_unimodal_univision_0723_3.npy', np.array(pred_scores))

        # return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]
        metrics, report_dict = self.processor.metric(true_labels, pred_labels)
        # roc_draw(true_labels, pred_scores, self.config.output_path + '/' + self.config.model_type + '/roc_curve.jpg')
        return metrics, report_dict


