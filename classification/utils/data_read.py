import os
import random
import torch
import pandas as pd
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from emr_process import EMR_FEATURES, only_29dim, one_hot
import pickle
import numpy as np
import time
# print(os.getcwd())

'''
1、特征向量的读取
2、划分训练集和测试集
3、另存为数据文件，需要将单模态与多模态的数据划分应该一致，即单模态的各部分的id应该与多模态的各部分的id一致
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
            name1 = name.split('_')[1]
            name2 = name.split('_')[2]
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
                id = '_'.join([name1, name2, str(i + 1)])
                case = {'id': id, 'tensor': tensor[i], 'label': label}
                tensor_list.append(case)

    print(len(tensor_list))
    print("benign:", benign_num, "malignant:", malignant_num)   #benign: 0 malignant: 3693
    print("读取完成")

    return tensor_list

def read_emr(labelfile, tensor_path, emr_path):
    emr_list = []
    id_list = []
    df_label = pd.read_csv(labelfile)
    emr_df = pd.read_csv(emr_path)
    # only_29dim(emr_df.loc[:, EMR_FEATURES[1: -1]])
    emr_df_ = one_hot(emr_df.loc[:, EMR_FEATURES[1: -1]])
    

    benign_num = 0
    malignant_num = 0

    for root, dirs, files in os.walk(tensor_path):
        for file in files:
            name = re.match(r'^[^\.]+', file).group(0)  #benign_S0000004_1
            name1 = name.split('_')[1]  #S0000004
            name2 = name.split('_')[2]  
            if df_label.loc[df_label['slide_id'] == name, 'label'].values == 'normal_tissue':
                label = 'benign'
                benign_num += 1
            else:
                label = 'malignant'
                malignant_num += 1
            tensor = torch.load(os.path.join(root, file))

            emr = emr_df_.loc[emr_df['Patient ID'] == name1].values[0]

            for i in range(len(tensor)):
                id = '_'.join([name1, name2, str(i + 1)])
                case = {'id': id, 'tensor': torch.FloatTensor(emr), 'label': label}
                emr_list.append(case)
    print(len(emr_list))
    print("benign:", benign_num, "malignant:", malignant_num)
    print("读取完成")
    return emr_list

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



def read_tensor_emr(labelfile, tensor_path, emr_path):

    tensor_emr_list = []
    id_list = []
    df_label = pd.read_csv(labelfile)
    emr_df = pd.read_csv(emr_path)
    # only_29dim(emr_df.loc[:, EMR_FEATURES[1: -1]])
    emr_df_ = one_hot(emr_df.loc[:, EMR_FEATURES[1: -1]])
    kg_embeddings = pickle.load(open('./data/patient_embeddings.pkl', 'rb'))
    kg_zero = np.zeros(512, np.float32)

    benign_num = 0
    malignant_num = 0

    for root, dirs, files in os.walk(tensor_path):
        for file in files:
            name = re.match(r'^[^\.]+', file).group(0)  #benign_S0000004_1
            name1 = name.split('_')[1]  #S0000004
            name2 = name.split('_')[2]  
            if df_label.loc[df_label['slide_id'] == name, 'label'].values == 'normal_tissue':
                label = 'benign'
                benign_num += 1
            else:
                label = 'malignant'
                malignant_num += 1
            tensor = torch.load(os.path.join(root, file))
            # emr = emr_df.loc[emr_df['Patient ID'] == name1, EMR_FEATURES[1: -1]].values[0]
            #values为：[[2 2 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1 1 1 1 1 1 2 0 0 0]]

            emr = emr_df_.loc[emr_df['Patient ID'] == name1].values[0]

            # print(emr)
            '''
            [ True False False False  True False  True False False False False  True
                False False  True False False  True False False  True False False  True
                False False  True False False  True False  True False False False  True
                False  True False  True False False False  True False False  True False
                False False  True False False  True False False False False False False
                True False False False  True False False False  True False False False
                False False False  True False False False False False False  True False
                False  True False False False  True False False False False False  True
                False  True False False  True False False  True False False]
            '''
            # time.sleep(60)
            if name1 in kg_embeddings:
                kg = kg_embeddings[name1][0]
            else:
                kg = kg_zero

            for i in range(len(tensor)):
                id = '_'.join([name1, name2, str(i + 1)])
                case = {'id': id, 'tensor': tensor[i], 'emr': torch.FloatTensor(emr), 'kg': torch.FloatTensor(kg), 'label': label}
                '''
                # 似乎可以考虑使用元组，这样可以减少内存占用
                case = (id, tensor[i], torch.LongTensor(emr), label)
                '''
                tensor_emr_list.append(case)
    print(len(tensor_emr_list))
    # print(tensor_list[0])
    print("benign:", benign_num, "malignant:", malignant_num)
    print("读取完成")

    return tensor_emr_list
    '''
    [
    {'tensor': tensor([0.1110, 0.0138, 0.0240,  ..., 0.0090, 0.0068, 0.0405]), 
    'emr': tensor([2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0]), #或106维的向量
    'label': 'benign'}
    ]
    '''



# labelfile = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\Image_list_new.csv'
# tensor_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\Features_directory\pt_files'
# emr_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\EMR.csv'
# list = read_tensor_emr(labelfile, tensor_path, emr_path)
# print(list[0]['tensor'].dtype)
# print(list[0])

# with open('./classification/data/data_id.json', 'w') as wf:
#     json.dump(list2, wf, indent = 4) 
    #尝试将Tensor对象序列化为JSON时，会遇到错误TypeError: Object of type 'Tensor' is not JSON serializable。这是因为Tensor对象不是JSON序列化数据类型，所以无法直接写入JSON文件。
    #需要先将tensor转换为list
    #但是直接转换为list，会丢失精度，所以可以转换为字符串。并且json格式也比.pt格式占据的空间大很多


'''
可以只对id写入json，然后tensor，emr，这些，在载入内存之后，可以根据id去取tensor，emr等数据。
如果将id以及数据地址写入，在之后的话根据地址去io，会产生大量的io时间，不利于训练。
'''

'''
with open('./classification/data/data_id.json', 'r') as f:
    data_id = json.load(f)
    train_ratio = 0.7
    valid_ratio = 0.1
    test_ratio = 0.2
    train_num = int(len(data_id) * train_ratio)
    valid_num = int(len(data_id) * valid_ratio)
    test_num = int(len(data_id) * test_ratio)
    # assert train_num + valid_num +test_num == len(data_id)
    
    
    random.shuffle(data_id)
    train_data = []
    valid_data = []
    test_data = []
    for id in data_id:
        # print(id)
        if len(train_data) < train_num:
            train_data.append(id)
        elif len(valid_data) < valid_num:
            valid_data.append(id)
        else:
            test_data.append(id)
    
    print(len(train_data), '\n',len(valid_data), '\n', len(test_data))
    print('finish')

    with open('./classification/data/train_id_1.txt', 'w') as f:
        for id in tqdm(train_data, desc='-----------train data'):
            f.write(id['id'])
            f.write('\n')
    
    with open('./classification/data/valid_id_1.txt', 'w') as f:
        for id in tqdm(valid_data, desc='-----------valid data'):
            f.write(id['id'])
            f.write('\n')

    with open('./classification/data/test_id_1.txt', 'w') as f:
        for id in tqdm(test_data, desc = '-----------test data'):
            f.write(id['id'])
            f.write('\n')
'''


'''
后期还可以重新划分一下，按照患者来划分

'''

'''
with open('./classification/data/data_id.json', 'r', encoding = 'utf-8') as f:
    exclusionid = open('./classification/data/exclusiondata.txt', 'r', encoding = 'utf-8').readlines()
    for i in range(len(exclusionid)): exclusionid[i] = exclusionid[i].strip('\n')
    # print(exclusionid)
    data_id = json.load(f)
    
    random.shuffle(data_id)
    
    train_data = []
    valid_data = []
    test_data = []

    for id in data_id:
        x = id['id'].split('_')[0]
        if x not in exclusionid:
            train_data.append(id)
        else:
            test_data.append(id)
    
    validratio = 0.1
    split_point = int(len(train_data) * (1 - validratio))
    valid_data = train_data[split_point:]
    train_data = train_data[:split_point]

    print(len(train_data), '\n', len(valid_data), '\n', len(test_data))
    print('finish')

    with open('./classification/data/exclusion_train_id.txt', 'w') as f:
        for id in tqdm(train_data, desc='-----------train data'):
            f.write(id['id'])
            f.write('\n')
    
    with open('./classification/data/exclusion_valid_id.txt', 'w') as f:
        for id in tqdm(valid_data, desc='-----------valid data'):
            f.write(id['id'])
            f.write('\n')

    with open('./classification/data/exclusion_test_id.txt', 'w') as f:
        for id in tqdm(test_data, desc = '-----------test data'):
            f.write(id['id'])
            f.write('\n')
'''