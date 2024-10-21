'''
普通的常用工具
'''

import os
import json
import chardet
import torch
from tqdm import tqdm   #进度条库
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random

def split_dataset(path, train_ratio, valid_ratio, test_ratio):   #分割数据集
    '''还得重写'''
    data = os.listdir(path)
    assert train_ratio + valid_ratio + test_ratio == 1, 'Ratio error.'   #判断划分是否正确
    
    train_nums = int(len(data) * train_ratio)
    valid_nums = int(len(data) * valid_ratio)
    test_nums = int(len(data) * test_ratio)
    
    train, valid, test = [], [], []
    for root, dirs, files in os.walk(path):
        random.shuffle(files)   #打乱文件顺序
        for file in files:
            if len(train) < train_nums:
                train.append(os.path.join(root, file))
            elif len(valid) < valid_nums:
                valid.append(os.path.join(root, file))
            else:
                test.append(os.path.join(root, file))
    return train, valid, test

def data_format(input_path, data_dir, output_path):
    data = []

def read_from_file():
    pass

def write_to_file(path, outputs):   #可以用来输出测试结果
    with open(path, 'w') as f:
        for line in tqdm(outputs, desc='----- [Writing]'):
            f.write(line)
            f.write('\n')
        f.close()

def save_model(output_path, model_type, model):
    '''可以用来输出训练后的模型'''
    output_model_dir = os.path.join(output_path, model_type)    #输出模型的保存目录
    if not os.path.exists(output_model_dir): os.makedirs(output_model_dir)
    model_to_save = model.module if hasattr(model, 'module') else model     # Only save the model it-self
    output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))