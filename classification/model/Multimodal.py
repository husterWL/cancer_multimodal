'''
加入电子病历数据，提高模型性能
'''
import torch
import torch.nn as nn
import torchvision.models as models



class Bicrossmodel(nn.Module):

    '''
    双向交叉注意力模型
    '''

    def __init__(self, config):
        
        super(Bicrossmodel, self).__init__()
        
        '''
        两个模态的特征首先需要对齐才能使用attention进行融合
        若使用concatenate的话，直接拼接即可，不需要对齐
        1、对两个模态进行线性变换，使其具有相同的embed_dim（如512）。
        2、将它们作为序列输入到多头注意力中，应用自注意力或交叉注意力。
        3、设置num_heads，并确保embed_dim能被num_heads整除。
        4、处理注意力输出，进行后续任务。
        '''

        '''
        存在的问题：
        模态1的信息丰富，样本数量多，而模态2的信息较少且有缺失，样本数量少，这可能意味着模态2的数据可能存在稀疏性或噪声较多。此外，模态2对模态1是一对多的关系，可能需要考虑如何有效地利用模态1的信息来增强模态2，或者反过来。
        
        模态1的信息更丰富，可能包含更多有用的特征，而模态2的信息较少且有零元素，可能存在缺失或噪声。
        如果任务目标是以模态1为主，比如用模态1的信息来补充模态2，那么可能需要将模态2作为Q，模态1作为K/V，这样模态2可以通过查询模态1的信息来增强自身。这也不太可能有效，因为这种情况下模态2的数据很少，很多K/V对应的Q是一样的，
        反之，如果任务需要模态2的信息来辅助模态1，则可能将模态1作为Q，模态2作为K/V，但这可能不太有效，因为模态2的信息不够丰富。
        
        模态2作为Q，模态1作为K/V​：
            优点：利用模态1的丰富信息来增强模态2，补充其稀疏性。
            缺点：如果模态2的样本数量少，可能导致模型无法充分学习到有效、有意义的查询方式。
        ​模态1作为Q，模态2作为K/V​：
            优点：模态1作为主导，利用模态2的辅助信息，但模态2的信息有限，可能效果不佳。
            缺点：模态2的稀疏性可能引入噪声，影响模态1的表示。模态2的数据是相同的，可能会导致所有patch的注意力权重相似，无法区分不同区域的重要性。
        ​双向交叉注意力​：
            优点：两个模态互相增强，信息交互更充分。
            缺点：模型复杂度高，需要更多数据，模态2的样本不足可能导致训练困难。
        self-attention：
            映射之后拼接两个模态，然后使用自注意力机制学习。

        需要处理模态2的稀疏性以及样本少的问题：
            降噪：进行去噪（如用非零均值填充零值）或使用 Dropout 层缓解过拟合    
            数据增强：进行轻微扰动（如添加高斯噪声）或插值生成新样本。
            还可以进行维度对齐，对数据升维
            ！图神经网络的表征的加入

            ​引入模态2的变换或生成不同表示​：如果模态2的数据是固定的，可能需要通过某种方式生成不同的表示来匹配不同的patch。

        进一步要考虑的问题：
            需要进一步考虑如何让模态2的信息能够差异化地影响每个patch，尤其是在模态2数据单一的情况下。可能需要通过模态2生成动态的权重或偏置，应用到各个patch的特征上，或者使用模态2作为Key/Value，让每个patch的Query去选择相关的信息。
            可能需要使用多头注意力来捕捉多方面的关系，或者引入残差连接来保留原始特征。  enhanced_patches = Q + attn_output
        '''
        

        self.fuse_attention = nn.MultiheadAttention(
            embed_dim = config.fusion_hidden_dimension, 
            num_heads = config.num_heads,
            batch_first = True,
            dropout = config.attention_dropout
            )

        # self.trans_attention = nn.TransformerEncoderLayer(
        #     d_model = config.fusion_hidden_dimension,
        # )

        self.modality_proj_img = nn.Linear(config.img_dimension, config.fusion_hidden_dimension)
        self.modality_proj_emr = nn.Linear(config.emr_dimension, config.fusion_hidden_dimension)    #后续可以尝试加入高斯噪声

        self.classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_dimension, config.output_hidden_dimension),
            nn.ReLU(inplace = True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.output_hidden_dimension, config.num_labels)
        )

        self.loss_func = nn.CrossEntropyLoss()
        
    def forward(self, tensors, emrs, labels = None):
        
        # print('weight_dtype', self.modality_proj_img.weight.dtype, '\n')
        # print('weight_dtype', self.modality_proj_emr.weight.dtype, '\n')
        # print(tensors.dtype, emrs.dtype)

        aligned_img = self.modality_proj_img(tensors)
        aligned_emr = self.modality_proj_emr(emrs)

        focused_img, _ = self.fuse_attention(aligned_emr, aligned_img, aligned_img)
        focused_emr, _ = self.fuse_attention(aligned_img, aligned_emr, aligned_emr)

        fused_feature = torch.cat([focused_img, focused_emr], dim = -1)

        prob_logits = self.classifier(fused_feature)
        pred_labels = torch.argmax(prob_logits, dim = 1)

        if labels is not None:
            loss = self.loss_func(prob_logits, labels)
            return pred_labels, loss
        else:
            return pred_labels
        
class Concatmodel(nn.Module):
    
    def __init__(self, config):

        super(Concatmodel, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(config.first_dropout),
            nn.Linear(config.middle_hidden_dimension, config.output_hidden_dimension),
            nn.ReLU(inplace = True),
            nn.Dropout(config.last_dropout),
            nn.Linear(config.output_hidden_dimension, config.num_labels)
        )

        self.modality_proj_img = nn.Linear(config.img_dimension, config.fusion_hidden_dimension)
        self.modality_proj_emr = nn.Linear(config.emr_dimension, config.fusion_hidden_dimension)

        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, tensors, emrs, labels = None):

        aligned_img = self.modality_proj_img(tensors)
        aligned_emr = self.modality_proj_emr(emrs)
        fused_feature = torch.cat([aligned_img, aligned_emr], dim = -1)
        
        prob_logits = self.classifier(fused_feature)
        pred_labels = torch.argmax(prob_logits, dim = 1)

        if labels is not None:
            loss = self.loss_func(prob_logits, labels)
            return pred_labels, loss
        else:
            return pred_labels
        
class GNN_Based(nn.Module):
    def __init__(self, config):
        super(GNN_Based, self).__init__()

    def forward():
        pass