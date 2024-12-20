'''
Dataset api: 与api_encode配合, 将api_encode的返回结果构造成Dataset方便Pytorch调用
    tips:
        注意如果数据长度不一需要编写collate_fn函数, 若无则将collate_fn设为None          ♥♥♥♥♥♥♥♥♥♥
'''

import numpy as np
import torch
from torch.utils.data import Dataset


class apidataset(Dataset):

    def __init__(self, guids, EHRs, imgs, labels):
        super().__init__()
        self.guids = guids
        self.EHRs = EHRs
        self.imgs = imgs
        self.labels = labels
    
    def __len__(self):
        return len(self.guids)  #返回有多少个样本
    
    def __getitem__(self, index):
        return self.guids[index], self.EHRs[index], self.imgs[index], self.labels[index]
    
    def collate_fn(self, batch):  #用于将数据集中的每个样本转换为一个批次，以便在训练和测试过程中使用。
        guids, EHRs, imgs, labels = map(list, zip(*batch))
        return guids, torch.tensor(EHRs).long(), torch.tensor(imgs).float(), torch.tensor(labels).long()