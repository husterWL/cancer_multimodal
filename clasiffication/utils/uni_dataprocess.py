import os
import random
import torch
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json

'''
1、特征向量的读取
2、划分训练集和测试集
3、
'''

# tensor_cancer = torch.load(
#     r"D:\BaiduNetdiskDownload\multimodal_breast_cancer\Features_directory\pt_files\benign_S0000004_1.pt")

# print(tensor_cancer.shape)  # 输出：torch.Size([patchs_num, 1048])
# print(tensor_cancer[1])


def read_tensor(labelfile, tensor_path):
    tensor_list = []
    df_label = pd.read_csv(labelfile)
    # print(df_label.head())
    benign_num = 0
    malignant_num = 0

    for root, dirs, files in os.walk(tensor_path):
        for file in files:
            name = re.match(r'^[^\.]+', file).group(0)
            # print(name)     #python unimodal_main.py --do_train
            if df_label.loc[df_label['slide_id'] == name, 'label'].values == 'normal_tissue':   #逻辑错误
                '''这一行的逻辑是错误的，因为 name in df_label.loc[df_label['slide_id'] == name, 'label'].values 
                这个表达式的结果是一个布尔值（True 或 False）'''
                label = 'benign'
                benign_num += 1
            else:
                label = 'malignant'
                malignant_num += 1
            tensor = torch.load(os.path.join(root, file))
            # print(len(tensor))  # 输出：torch.Size([patchs_num, 1048])
            for i in range(len(tensor)):
                case = {'tensor': tensor[i], 'label': label}
                tensor_list.append(case)

    print(len(tensor_list))
    # print(tensor_list[0])
    print("benign:", benign_num, "malignant:", malignant_num)   #benign: 0 malignant: 3693
    print("读取完成")

    return tensor_list

def split_dataset(data, train_ratio, valid_ratio, test_ratio):   #分割数据集
    assert train_ratio + valid_ratio + test_ratio == 1, 'Ratio error.'   #判断划分是否正确
    
    train_nums = int(len(data) * train_ratio)
    valid_nums = int(len(data) * valid_ratio)
    test_nums = int(len(data) * test_ratio)
    
    train, valid, test = [], [], []
    random.shuffle(data)
    for tensor in data:
        if len(train) < train_nums:
            train.append(tensor)
        elif len(valid) < valid_nums:
            valid.append(tensor)
        else:
            test.append(tensor)
    
    return train, valid, test

def get_loader(data, batch_size):
    dataset = Dataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

