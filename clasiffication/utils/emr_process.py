import pandas as pd
import numpy as np

EMR_FEATURES = [ 'Patient ID', 
                 'Age', 'Gender', 'Disease Course Type', 'Personal Tumor History', 'Family Tumor History',
                 'Prophase Treatment', 'Neoadjuvant Chemotherapy', 'Dimple Sign', 'Orange Peel Appearance',
                 'Redness And Swelling Of Skin', 'Skin Ulcers', 'Tumor', 'Breast Deformation', 'Nipple Change',
                 'Nipple Discharge', 'Axillary Lymphadenectasis', 'Swelling Of Lymph Nodes', 'Tumor Position', 
                 'Tumor Number', 'Tumor Size', 'Tumor Texture', 'Tumor Border', 'Smooth Surface', 'Tumor Morphology',
                 'Activity', 'Capsules', 'Tenderness', 'Skin Adhesion', 'Pectoral Muscle Adhesion', 
                 'Diagnosis_Belnign_1_Malignant_2']

# emr_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\EMR.csv'
# df = pd.read_csv(emr_path)
# print(df.head())

# feature = df.loc[ : , EMR_FEATURES[1: -1]]
# new = feature.copy()

# print(new['Age'].value_counts())
# print(new['Age'].describe())

# for i in range(len(EMR_FEATURES[1:-1])):
    # print(new[EMR_FEATURES[i+1]].value_counts())

    #两个想法：
    #1. 就只使用这29个维度，将缺失值先改为na，再将其他有0和na的维度全+1，再将na改为0，这样，0就是缺失值。之后加入到图像的tensor中
    #2. 将每个维度转换为独热编码，缺失值用0表示，其他用1表示。形成一个新的张量并和图像的tensor合并

    # 升维问题，可以考虑下自动编码器和GAN


def only_29dim(dataframe):

    dataframe.replace(-1, np.nan, inplace=True)
    columns_with_zero = [col for col in dataframe.columns if dataframe[col].isin([0]).any()]
    dataframe[columns_with_zero] = dataframe[columns_with_zero].apply(lambda x: x + 1).fillna(0)

    return dataframe    #29dims

def one_hot(dataframe):
    
    dataframe.replace(-1, np.nan, inplace = True)
    dataframe[ : ] = dataframe[ : ].apply(lambda col: pd.Categorical(col))
    dataframe = pd.get_dummies(dataframe, prefix_sep = '_', dummy_na = True, drop_first = False)
    
    return dataframe    #106dims

# for i in range(len(EMR_FEATURES[1:-1])):
#     print(new[EMR_FEATURES[i+1]].value_counts())

# only_29dim(df.loc[ : , EMR_FEATURES[1: -1]])
# print(df.loc[df['Patient ID'] == 'S0000004', EMR_FEATURES[1: -1]].values)

#大模型的扩充
#使用TITAN对每一张WSI进行报告的生成