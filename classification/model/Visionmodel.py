import timm
from .timm_wrapper import TimmCNNEncoder
# import torch



def get_encoder(model_name):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()

    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    return model


# model = torch.load('/mnt/Model/resnet50_tv_in1k/model.safetensors')
# model = torch.load('/mnt/Model/resnet50_tv_in1k/pytorch_model.bin')
# print(model)
