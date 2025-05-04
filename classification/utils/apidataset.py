'''
Dataset api: 与api_encode配合, 将api_encode的返回结果构造成Dataset方便Pytorch调用
    tips:
        注意如果数据长度不一需要编写collate_fn函数, 若无则将collate_fn设为None          ♥♥♥♥♥♥♥♥♥♥
'''

import numpy as np
import torch
from torch.utils.data import Dataset


class apidataset(Dataset):

    def __init__(self, guids, imgs, EHRs, labels):
        super().__init__()
        self.guids = guids
        self.imgs = imgs
        self.EHRs = EHRs
        self.labels = labels
    
    def __len__(self):
        return len(self.guids)  #返回有多少个样本
    
    def __getitem__(self, index):
        return self.guids[index], self.imgs[index], self.EHRs[index], self.labels[index]
    
    '''
    有问题的collate_fn函数
    def collate_fn(self, batch):  #用于将数据集中的每个样本转换为一个批次，以便在训练和测试过程中使用。
        guids, EHRs, imgs, labels = map(list, zip(*batch))
        return torch.tensor(EHRs).long(), torch.tensor(imgs).float(), torch.tensor(labels).long()
    '''

    def collate_fn(self, batch):  #用于将数据集中的每个样本转换为一个批次，以便在训练和测试过程中使用。
        
        guids = [b[0] for b in batch]
        imgs  = [b[1] for b in batch]
        imgs = torch.stack(imgs, 0)
        ehrs = [b[2] for b in batch]
        ehrs = torch.stack(ehrs)
        labels = torch.LongTensor([b[3] for b in batch])

        return guids, ehrs, imgs, labels
    
class uniapidataset(Dataset):

    def __init__(self, tensors, labels):
        super().__init__()
        self.tensors = tensors
        self.labels = labels
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, index):
        return self.tensors[index], self.labels[index]
    
    def collate_fn(self, batch):
        # tensors, labels = map(list, zip(*batch))
        # return torch.tensor(tensors).float(), torch.tensor(labels).long()
        # print((b[0] for b in batch))
        # tensors = (b[0] for b in batch)
        # labels = torch.LongTensor(b[1] for b in batch)

        # def collate_fn(self, batch):
        #     tensors = [b[0] for b in batch]
        #     labels = torch.LongTensor([b[1] for b in batch])
        #     return tensors, labels


        # for b in batch:
        #     tensors = [b[0]]
        #     labels = torch.LongTensor([b[1]])
        
        tensors = [b[0] for b in batch]
        tensors = torch.stack(tensors)
        labels = torch.LongTensor([b[1] for b in batch])

        # print(type(tensors), type(labels))
        return tensors, labels