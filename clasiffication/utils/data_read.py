import os
import random
import torch
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from emr_process import EMR_FEATURES, only_29dim, one_hot

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
            # print(len(tensor))  # 输出：torch.Size([patchs_num, 1024])
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

def read_tensor_emr(labelfile, tensor_path, emr_path):
    tensor_emr_list = []
    df_label = pd.read_csv(labelfile)
    emr_df = pd.read_csv(emr_path)
    only_29dim(emr_df.loc[:, EMR_FEATURES[1: -1]])
    

    benign_num = 0
    malignant_num = 0

    for root, dirs, files in os.walk(tensor_path):
        for file in files:
            name = re.match(r'^[^\.]+', file).group(0)  #benign_S0000004_1
            name1 = name.split('_')[1]  #S0000004
            if df_label.loc[df_label['slide_id'] == name, 'label'].values == 'normal_tissue':
                label = 'benign'
                benign_num += 1
            else:
                label = 'malignant'
                malignant_num += 1
            tensor = torch.load(os.path.join(root, file))
            emr = emr_df.loc[emr_df['Patient ID'] == name1, EMR_FEATURES[1: -1]].values[0]
            #vaalues为：[[2 2 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1 1 1 1 1 1 2 0 0 0]]
            for i in range(len(tensor)):
                case = {'tensor': tensor[i], 'emr': torch.LongTensor(emr), 'label': label}
                tensor_emr_list.append(case)

    print(len(tensor_emr_list))
    # print(tensor_list[0])
    print("benign:", benign_num, "malignant:", malignant_num)   #benign: 0 malignant: 3693
    print("读取完成")

    return tensor_emr_list
    '''
    [
    {'tensor': tensor([0.1110, 0.0138, 0.0240,  ..., 0.0090, 0.0068, 0.0405]), 
    'emr': tensor([2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0]), 
    'label': 'benign'}
    ]
    '''


# labelfile = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\Image_list_new.csv'
# tensor_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\Features_directory\pt_files'
# emr_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\EMR.csv'
# list = read_tensor_emr(labelfile, tensor_path, emr_path)
# print(list[0: 5])