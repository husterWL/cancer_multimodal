'''
使用单一的视觉进行良恶性分类
'''
import torch
import torch.nn as nn
import torchvision.models as models

# FCN = models.segmentation.fcn_resnet50(pretrained = True)


class Univision(nn.Module):

    def __init__(self, config):
        super(Univision, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.first_dropout),
            nn.Linear(config.middle_hidden_dimension, config.output_hidden_dimension),
            nn.ReLU(inplace = True),
            nn.Dropout(config.last_dropout),
            nn.Linear(config.output_hidden_dimension, config.num_labels),
            # nn.Softmax()
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, tensors, labels = None):
        features = tensors
        prob_logits = self.classifier(features)
        pred_labels = torch.argmax(prob_logits, dim = 1)

        if labels is not None:  #train、valid、test
            loss = self.loss_func(prob_logits, labels)
            return pred_labels, loss
        else:
            return pred_labels, prob_logits[:, 1]

class Uniemr(nn.Module):

    def __init__(self, config):
        super(Uniemr, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(config.first_dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace = True),
            nn.Dropout(config.last_dropout),
            nn.Linear(256, config.num_labels),
            # nn.Softmax(dim = 1)
        )

        self.modality_proj_emr = nn.Linear(config.emr_dimension, config.fusion_hidden_dimension)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, emrs, labels = None):
        
        features = self.modality_proj_emr(emrs)
        prob_logits = self.classifier(features)
        pred_labels = torch.argmax(prob_logits, dim = 1)

        if labels is not None:
            loss = self.loss_func(prob_logits, labels)
            return pred_labels, loss
        else:
            return pred_labels, prob_logits[:, 1]

class Univision_sa(nn.Module):  #效果非常差   毕竟不是序列数据
    
    def __init__(self, config):
        super(Univision_sa, self).__init__()

        self.selfattention = nn.MultiheadAttention(
            embed_dim = config.img_dimension,
            num_heads = config.num_heads,
            batch_first = True,
            dropout = config.attention_dropout
        )

        self.classifier = nn.Sequential(
            nn.Dropout(config.first_dropout),
            nn.Linear(config.middle_hidden_dimension, config.output_hidden_dimension),
            nn.ReLU(inplace = True),
            nn.Dropout(config.last_dropout),
            nn.Linear(config.output_hidden_dimension, config.num_labels),
            # nn.Softmax(dim = 1)
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, tensors, labels = None):
        
        # print(tensors.shape) (16, 1024)
        tensors = tensors.unsqueeze(1)
        sa_imgtensors, _ = self.selfattention(tensors, tensors, tensors)
        sa_imgtensors = sa_imgtensors.squeeze(1)
        prob_logits = self.classifier(sa_imgtensors)
        pred_labels = torch.argmax(prob_logits, dim = 1)

        if labels is not None:
            loss = self.loss_func(prob_logits, labels)
            return pred_labels, loss
        else:
            return pred_labels, prob_logits[:, 1]