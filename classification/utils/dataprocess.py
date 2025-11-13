'''
data process: 数据处理, 包括 标签Vocab 和 数据处理类
    tips:
        其中标签Vocab实例化对象必须在api_encode中被调用(add_label)
'''

from typing import Any
from torch.utils.data import DataLoader
'''
这里需要导入encode、decode、metric以及dataset的API
'''

from apidataset import apidataset
from apidataset import uniapidataset
from apiencode import api_encode
from apidecode import api_decode
from apimetric import api_metric
from tqdm import tqdm

class LabelVocabulary():  #类的作用：将标签名(str)映射为整数(value)值
    UNK = 'UNK'

    def __init__(self):
        self.label2value = {}   #benign, malignant
        self.value2label = {}

    def _length_(self):
        return len(self.label2value)
    
    def add_label(self, label):
        if label not in self.label2value:
            self.label2value.update({label: len(self.label2value)})   #benign, malignant : 0/1(相应的value)   
            self.value2label.update({len(self.value2label): label})
    
    def label_to_value(self, label):
        return self.label2value.get(label)
    
    def value_to_label(self, value):
        return self.value2label.get(value) 

class Processor():

    def __init__(self, config) -> None: #->后面是静态注释，在._annotations_中，参数的注释是使用冒号:
        self.config = config
        self.labelvocab = LabelVocabulary()

    def __call__(self, data, parameters):
        return self.to_loader(data, parameters)
    
    def encode(self, data):
        return api_encode(data, self.labelvocab, self.config)
    
    def decode(self, outputs):
        return api_decode(outputs, self.labelvocab)

    def metric(self, inputs, outputs):
        return api_metric(inputs, outputs)
    
    def to_dataset(self, data):
        dataset_inputs = self.encode(data)  # 返回的是多个值，多个值付给dataset_inputs是可以的，python中是利用元组打包形式进行赋值
        # 无论是否显式添加括号，逗号分隔的多个值在 Python 中默认被视为元组
        return apidataset(*dataset_inputs)
    
    def to_loader(self, data, parameters):
        dataset = self.to_dataset(data)
        return DataLoader(dataset = dataset, **parameters, collate_fn = dataset.collate_fn)

class Uni_processor():
    def __init__(self, config) -> None:
        self.config = config
        self.labelvocab = LabelVocabulary()
    
    def __call__(self, data, parameters):
        return self.to_loader(data, parameters) 
        #返回的是一个DataLoader对象，data的格式为：[{'tensor': tensor, 'label': label}, ...]

    def encode(self, data):
        self.labelvocab.add_label('benign')
        self.labelvocab.add_label('malignant')
        # print('这是labelvocab的长度', self.labelvocab._length_()) #没问题 为2
        # print('这是labelvocab的value2label', self.labelvocab.label2value) #没问题 {'benign': 0, 'malignant': 1}
        
        guids, tensors, encoded_labels = [], [], []
        for line in tqdm(data, desc='----- [Encoding]'):
            guid, tensor, label = line['id'], line['tensor'], line['label']
            guids.append(guid)
            tensors.append(tensor)
            encoded_labels.append(self.labelvocab.label_to_value(label))

        # print(guids[0])
        # print(tensors[0])
        # print(tensors[0].shape)
        # print(encoded_labels[0])
        return guids, tensors, encoded_labels
    
    def metric(self, inputs, outputs):
        return api_metric(inputs, outputs)
    
    def to_dataset(self, data):
        dataset_inputs = self.encode(data)
        return uniapidataset(*dataset_inputs)

    def to_loader(self, data, parameters):
        dataset = self.to_dataset(data)
        return DataLoader(dataset = dataset, **parameters, collate_fn = dataset.collate_fn)

class wsi_patch_dataset():
    def __init__(self, config) -> None:
        self.config = config
        self.labelvocab = LabelVocabulary()
    
    def __call__(self, data, parameters) -> Any:
        
        self.labelvocab.add_label('benign')
        self.labelvocab.add_label('malignant')

        if not self.config.model_type == 'unimodal':
            guids, encoded_imgs, encoded_EHRs, encoded_KGs, encoded_labels = [], [], [], [], []
            for line in tqdm(data, desc='----- [Encoding]'):
                guid, img, emr, kg, label = line['id'], line['tensor'], line['emr'], line['kg'], line['label']

                guids.append(guid)
                encoded_imgs.append(img)
                encoded_EHRs.append(emr)
                encoded_KGs.append(kg)
                encoded_labels.append(self.labelvocab.label_to_value(label))
        
        elif self.config.fusion_type == 'unimodal' or self.config.fusion_type == 'Univision_sa':
            guids, tensors, encoded_labels = [], [], []
            for line in tqdm(data, desc='----- [Encoding]'):
                guid, tensor, label = line['id'], line['tensor'], line['label']
                guids.append(guid)
                tensors.append(tensor)
                encoded_labels.append(self.labelvocab.label_to_value(label))