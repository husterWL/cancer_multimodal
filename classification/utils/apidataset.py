'''
Dataset api: 与api_encode配合, 将api_encode的返回结果构造成Dataset方便Pytorch调用
    tips:
        注意如果数据长度不一需要编写collate_fn函数, 若无则将collate_fn设为None          ♥♥♥♥♥♥♥♥♥♥
'''

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class apidataset(Dataset):

    def __init__(self, guids, imgs, EHRs, KGs, labels):
        super().__init__()
        self.guids = guids
        self.imgs = imgs
        self.ehrs = EHRs
        self.kgs =  KGs
        self.labels = labels
    
    def __len__(self):
        return len(self.guids)  #返回有多少个样本
    
    def __getitem__(self, index):
        return self.guids[index], self.imgs[index], self.ehrs[index], self.kgs[index], self.labels[index]
    
    '''
    有问题的collate_fn函数
    def collate_fn(self, batch):  #用于将数据集中的每个样本转换为一个批次，以便在训练和测试过程中使用。
        guids, EHRs, imgs, labels = map(list, zip(*batch))
        return torch.tensor(EHRs).long(), torch.tensor(imgs).float(), torch.tensor(labels).long()
    '''

    def collate_fn(self, batch):  #用于将数据集中的每个样本转换为一个批次，以便在训练和测试过程中使用。
        
        guids = [b[0] for b in batch]
        imgs = [img_transforms(Image.open(b[1]).convert('RGB')) for b in batch]
        # imgs = [b[1] for b in batch]
        imgs = torch.stack(imgs)
        ehrs = [b[2] for b in batch]
        ehrs = torch.stack(ehrs)
        kgs = [b[3] for b in batch]
        kgs = torch.stack(kgs)
        labels = torch.LongTensor([b[4] for b in batch])


        return guids, imgs, ehrs, kgs, labels
    
class uniapidataset(Dataset):

    def __init__(self, guids, tensors, labels):
        super().__init__()
        self.guids = guids
        self.tensors = tensors
        self.labels = labels
    
    def __len__(self):
        return len(self.guids)
    
    def __getitem__(self, index):
        return self.guids[index], self.tensors[index], self.labels[index]
    
    def collate_fn(self, batch):
        
        guids = [b[0] for b in batch]
        tensors = [img_transforms(Image.open(b[1]).convert('RGB')) for b in batch]
        # tensors = [b[1] for b in batch]
        tensors = torch.stack(tensors)
        labels = torch.LongTensor([b[2] for b in batch])

        # print(type(tensors), type(labels))
        return guids, tensors, labels