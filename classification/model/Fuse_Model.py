'''
thought
using multihead_attention to fuse image and structured data.

for image and EHRs, maybe we can use GANs
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import config
# from transformers import AutoModel
from torchvision import models

class FuseModel_1(nn.Module): #融合模型还需要看教材与project
    pass
    def __init__(self, config):
        super(FuseModel_1, self).__init__()
        self.config = config

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2)
        )

    def forward(self, tensors, emrs, labels = None):
        pass
        fusefeature = torch.cat((tensors, emrs), dim = -1)
        
        prob_vector = self.classifier(fusefeature)
        pred_labels = torch.argmax(prob_vector, dim = 1)

        if labels is not None:  #train、valid、test
            loss = self.loss_func(prob_vector, labels)
            return pred_labels, loss
        else:
            return pred_labels  #predict

class FuseModel_2(nn.Module):
    pass
    def __init__(self, config):
        super(FuseModel_2, self).__init__()
        self.config = config

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2)
        )

        self.fuseattention = nn.MultiheadAttention(
            embed_dim = config.hidden_size, 
            num_heads = 1,
            dropout = config.attention_dropout
            )

    def forward(self, tensors, emrs, labels = None):
        pass
        fuseweights, _ = self.fuseattention(tensors, emrs, emrs, need_weights = False)
        fusefeature = tensors * fuseweights
        prob_vector = self.classifier(fusefeature)
        pred_labels = torch.argmax(prob_vector, dim = 1)

        if labels is not None:  #train、valid、test
            loss = self.loss_func(prob_vector, labels)
            return pred_labels, loss
        else:
            return pred_labels  #predict
        