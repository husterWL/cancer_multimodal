'''
使用单一的视觉进行良恶性分类
'''
import torch
import torch.nn as nn
from Config import config
import torchvision.models as models

class VisionModel(nn.Module):

    def __init__(self):
        super(VisionModel, self).__init__()
        # 使用预训练的resnet101模型，删除最后的3层
        self.features = models.resnet101(pretrained=True)
        self.features.fc = nn.Sequential()

    def forward(self, images):
        x = self.features(images)
        return x
    
class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        # 使用两个全连接层进行分类
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, config.num_classes)
        # 使用relu和dropout作为激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, features):
        x = features
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x