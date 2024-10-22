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
