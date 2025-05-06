import torch
import torch.nn as nn
from Config import config
import torchvision.models as models

'''
加入电子病历数据，提高模型性能
'''

class Fusemodel(nn.Module):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        

        '''
        两个模态的特征首先需要对齐才能使用attention进行融合
        若使用concatenate的话，直接拼接即可，不需要对齐
        1、对两个模态进行线性变换，使其具有相同的embed_dim（如512）。
        2、将它们作为序列输入到多头注意力中，应用自注意力或交叉注意力。
        3、设置num_heads，并确保embed_dim能被num_heads整除。
        4、处理注意力输出，进行后续任务。
        '''
        

        self.fuse_attention = nn.MultiheadAttention(
            embed_dim = config.fusion_hidden_dimension, 
            num_heads = config.num_heads,
            batch_first = True
            )

        self.modality_proj_img = nn.Linear(config.img_dimension, config.fusion_hidden_dimension)
        self.modality_proj_emr = nn.Linear(config.emr_dimension, config.fusion_hidden_dimension)

        self.clasiffier = nn.Sequential(
            nn.Linear(config.fusion_hidden_dimension, config.fusion_hidden_dimension),
        )
        
    def forward(self, *args, **kwargs):
        pass