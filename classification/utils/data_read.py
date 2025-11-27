import os
import random
import torch
import pandas as pd
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from emr_process import EMR_FEATURES, only_29dim, one_hot, totext
import pickle
import numpy as np
import time
import h5py
# import openslide
from torchvision import transforms
from PIL import Image
'''
1、特征向量的读取
2、划分训练集和测试集
3、另存为数据文件，需要将单模态与多模态的数据划分应该一致，即单模态的各部分的id应该与多模态的各部分的id一致
'''


MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": [0.485, 0.456, 0.406],
		"std": [0.229, 0.224, 0.225]
	},

}

def get_eval_transforms(mean, std, target_img_size = -1):
	trsforms = []
	
	if target_img_size > 0:
		trsforms.append(transforms.Resize(target_img_size))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms

constants = MODEL2CONSTANTS['resnet50_trunc']
img_transforms = get_eval_transforms(
    mean = constants['mean'],
    std = constants['std'],
    target_img_size = 224,
)


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

def read_kg(labelfile, tensor_path):

    kg_list = []
    id_list = []
    df_label = pd.read_csv(labelfile)
    kg_embeddings = pickle.load(open('./data/patient_embeddings_3.pkl', 'rb'))
    kg_zero = np.zeros(512, np.float32)


    for root, dirs, files in os.walk(tensor_path):
        for file in files:
            name = re.match(r'^[^\.]+', file).group(0)  #benign_S0000004_1
            name1 = name.split('_')[1]  #S0000004
            name2 = name.split('_')[2]  
            if df_label.loc[df_label['slide_id'] == name, 'label'].values == 'normal_tissue':
                label = 'benign'

            else:
                label = 'malignant'

            tensor = torch.load(os.path.join(root, file))

            if name1 in kg_embeddings:
                kg = kg_embeddings[name1][0]
            else:
                kg = kg_zero

            for i in range(len(tensor)):
                id = '_'.join([name1, name2, str(i + 1)])
                case = {'id': id, 'tensor': torch.FloatTensor(kg), 'label': label}
                kg_list.append(case)
    return kg_list

def read_tensor_emr(labelfile, tensor_path, emr_path):

    tensor_emr_list = []
    id_list = []
    df_label = pd.read_csv(labelfile)
    emr_df = pd.read_csv(emr_path)
    # only_29dim(emr_df.loc[:, EMR_FEATURES[1: -1]])
    emr_df_ = one_hot(emr_df.loc[:, EMR_FEATURES[1: -1]])
    kg_embeddings = pickle.load(open('./data/patient_embeddings_3.pkl', 'rb'))
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

#2025年11月4日

def read_img(labelfile, coords_path, img_path):
    img_list = []
    df_label = pd.read_csv(labelfile)
    # print(df_label.head())
    benign_num = 0
    malignant_num = 0

    for root, dirs, files in os.walk(coords_path):
        for file in tqdm(files, desc = '------------------Processing files'):
            name = re.match(r'^[^\.]+', file).group(0)
            name1 = name.split('_')[1]
            name2 = name.split('_')[2]
            if df_label.loc[df_label['slide_id'] == name, 'label'].values == 'normal_tissue':   #逻辑错误
                label = 'benign'
                benign_num += 1
            else:
                label = 'malignant'
                malignant_num += 1
            with h5py.File(os.path.join(root, file), 'r') as f:
                # print(f['features'].shape)
                # print(f['features'][:])

                # print(f['coords'].shape)
                # print(f['coords'][:])
                coords = f['coords'][:]
                # patch_level = f['coords'].attrs['patch_level']
                # patch_size = f['coords'].attrs['patch_size']
                patch_level = 0
                patch_size = 256
            '''
            [   
                [   0    0]
                [   0  256]
                [   0  512]
                [   0  768]
                [   0 1024]
                [   0 1280]
                [ 256    0]
                [ 256  256]
                ....
            ]
            '''
            # print(len(tensor))  # 输出：torch.Size([patchs_num, 1024])
            # slide_file_path = os.path.join(img_path, name + '.tif')
            
            # for i in tqdm(range(len(coords)), desc = '-------------Reading patches from ' + slide_file_path):
            for i in range(len(coords)):
                id = '_'.join([name1, name2, str(i + 1)])
                #读取wsi病理图像
                patch_path = os.path.join(img_path, id + '.png')
                # if os.path.exists(save_path):
                #     print('skip: ' + slide_file_path)
                #     break

                # wsi = openslide.open_slide(slide_file_path)
                # img = wsi.read_region(coords[i], patch_level, (patch_size, patch_size)).convert('RGB')
                
                # img = Image.open(patch_path).convert('RGB')
                #保存用
                # img.save(save_path)

                # img = img_transforms(img)
                case = {'id': id, 'tensor': patch_path, 'label': label}
                img_list.append(case)

    print(len(img_list))
    print("benign:", benign_num, "malignant:", malignant_num)   #benign: 0 malignant: 3693
    print("读取完成")

    return img_list

def read_img_emr(labelfile, coords_path, img_path, emr_path):

    tensor_emr_list = []
    id_list = []
    df_label = pd.read_csv(labelfile)
    emr_df = pd.read_csv(emr_path)
    # only_29dim(emr_df.loc[:, EMR_FEATURES[1: -1]])
    emr_df_ = one_hot(emr_df.loc[:, EMR_FEATURES[1: -1]])
    kg_embeddings = pickle.load(open('./data/patient_embeddings_3.pkl', 'rb'))
    kg_zero = np.zeros(512, np.float32)

    benign_num = 0
    malignant_num = 0

    for root, dirs, files in os.walk(coords_path):
        for file in tqdm(files, desc = '------------------------[Processing files]'):
            name = re.match(r'^[^\.]+', file).group(0)  #benign_S0000004_1
            name1 = name.split('_')[1]  #S0000004
            name2 = name.split('_')[2]  
            if df_label.loc[df_label['slide_id'] == name, 'label'].values == 'normal_tissue':
                label = 'benign'
                benign_num += 1
            else:
                label = 'malignant'
                malignant_num += 1
            with h5py.File(os.path.join(root, file), 'r') as f:
                
                coords = f['coords'][:]
                # patch_level = f['coords'].attrs['patch_level']
                # patch_size = f['coords'].attrs['patch_size']
                patch_level = 0
                patch_size = 256
            
            # slide_file_path = os.path.join(img_path, name + '.tif')

            emr = emr_df_.loc[emr_df['Patient ID'] == name1].values[0]

            # print(emr)
            
            # time.sleep(60)
            if name1 in kg_embeddings:
                kg = kg_embeddings[name1][0]
            else:
                kg = kg_zero

            for i in range(len(coords)):
                id = '_'.join([name1, name2, str(i + 1)])
                patch_path = os.path.join(img_path, id + '.png')
                #读取wsi病理图像
                # wsi = openslide.open_slide(slide_file_path)
                # img = wsi.read_region(coords[i], patch_level, (patch_size, patch_size)).convert('RGB')
                # img = Image.open(patch_path).convert('RGB')
                # img = img_transforms(img)
                case = {'id': id, 'tensor': patch_path, 'emr': torch.FloatTensor(emr), 'kg': torch.FloatTensor(kg), 'label': label}
                
                tensor_emr_list.append(case)

    print(len(tensor_emr_list))

    print("benign:", benign_num, "malignant:", malignant_num)
    print("读取完成")

    return tensor_emr_list

# labelfile = '/mnt/Data/breast_cancer/Image_list_new.csv'
# coords_path = '/mnt/Data/breast_cancer/h5_files'
# img_path = '/mnt/Data/breast_cancer/image'
# output_path = '/mnt/Data/breast_cancer/patches'
# read_img(labelfile, coords_path, img_path, output_path)

def read_img_text(labelfile, coords_path, img_path, emr_path):

    tensor_text_list = []
    id_list = []
    df_label = pd.read_csv(labelfile)
    emr_df = pd.read_csv(emr_path)
    emr_df_ = totext(emr_df.loc[:, EMR_FEATURES[1: -1]])
    kg_embeddings = pickle.load(open('./data/patient_embeddings_3.pkl', 'rb'))
    kg_zero = np.zeros(512, np.float32)

    benign_num = 0
    malignant_num = 0

    for root, dirs, files in os.walk(coords_path):
        for file in tqdm(files, desc = '------------------------[Processing files]'):
            name = re.match(r'^[^\.]+', file).group(0)  #benign_S0000004_1
            name1 = name.split('_')[1]  #S0000004
            name2 = name.split('_')[2]  
            if df_label.loc[df_label['slide_id'] == name, 'label'].values == 'normal_tissue':
                label = 'benign'
                benign_num += 1
            else:
                label = 'malignant'
                malignant_num += 1
            with h5py.File(os.path.join(root, file), 'r') as f:
                
                coords = f['coords'][:]
                # patch_level = f['coords'].attrs['patch_level']
                # patch_size = f['coords'].attrs['patch_size']
                patch_level = 0
                patch_size = 256

            text = emr_df_.loc[emr_df['Patient ID'] == name1]['all'].values[0]

            if name1 in kg_embeddings:
                kg = kg_embeddings[name1][0]
            else:
                kg = kg_zero

            for i in range(len(coords)):
                id = '_'.join([name1, name2, str(i + 1)])
                patch_path = os.path.join(img_path, id + '.png')
                case = {'id': id, 'tensor': patch_path, 'emr': text, 'kg': torch.FloatTensor(kg), 'label': label}
                
                tensor_text_list.append(case)

    print(len(tensor_text_list))

    print("benign:", benign_num, "malignant:", malignant_num)
    print("读取完成")

    return tensor_text_list