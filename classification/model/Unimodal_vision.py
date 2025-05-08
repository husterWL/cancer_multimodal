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
        prob_vector = self.classifier(features)
        pred_labels = torch.argmax(prob_vector, dim = 1)

        if labels is not None:  #train、valid、test
            loss = self.loss_func(prob_vector, labels)
            return pred_labels, loss
        else:
            return pred_labels  #predict