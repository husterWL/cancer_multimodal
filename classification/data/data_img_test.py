import os
import re
import csv
import pandas as pd
import numpy as np
import pickle
import torch
import h5py
from PIL import Image
from torchvision import transforms
import torch
import timm
from safetensors.torch import load_file

class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k',
                 checkpoint_path: str = '/mnt/Model/resnet50_tv_in1k/model.safetensors',
                #  pretrained_cfg_overlay = dict(file = '/mnt/Model/resnet50_tv_in1k/pytorch_model.bin'),
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0},
                # kwargs: dict = {'features_only': True, 'out_indices': (3,), 'num_classes': 0},
                 pool: bool = True):
        super().__init__()
        assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        self.model = timm.create_model(model_name, **kwargs)
        # self.model = timm.create_model(model_name, checkpoint_path = checkpoint_path , pretrained_cfg_overlay = pretrained_cfg_overlay, **kwargs)
        # self.model = timm.create_model(model_name, checkpoint_path = checkpoint_path , **kwargs)


        self.model = self.model.eval()
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ptfile = r'F:\multimodal_breast_cancer\Features_directory\pt_files\benign_S0000004_1.pt'
h5file = r'F:\multimodal_breast_cancer\Features_directory\h5_files\benign_S0000004_1.h5'
coordfile = r'F:\multimodal_breast_cancer\Result_directory\patches\benign_S0000004_1.h5'

mntfiles = r'/mnt/Data/breast_cancer/h5_files/benign_S0000004_1.h5'
ptfiles = r'/mnt/Data/breast_cancer/pt_files/malignant_S0016018_17.pt'

# with open(ptfile, 'rb') as f:
#     pt = torch.load(f)
#     print(pt.shape)
#     print(pt)

# with h5py.File(h5file, 'r') as f:
#     print(f['features'].shape)
#     print(f['features'][:])

#     print(f['coords'].shape)
#     print(f['coords'][:])

# with h5py.File(mntfiles, 'r') as f:
#     print(f['coords'].shape)
#     print(f['coords'][:])
#     # print( f['coords'].attrs['patch_level'])
#     # print( f['coords'].attrs['patch_size'])
#     file = f['coords'][:]
#     print(file[32])

# tensor = torch.load(ptfiles)
# print(tensor[1])
# img_list = []
img = Image.open('/mnt/Data/breast_cancer/patches/S0004176_14_45.png').convert('RGB')
img = img_transforms(img)
# img_list.append(img)
img = img.unsqueeze(0)
model = TimmCNNEncoder()
print(model)

model1 = torch.nn.Module()
state_dict = torch.load('/mnt/Model/resnet50_tv_in1k/pytorch_model.bin', map_location = 'cpu', weights_only = True)
model1.load_state_dict(state_dict, strict = False)
print(model1)