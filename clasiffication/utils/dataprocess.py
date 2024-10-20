'''
data process: 数据处理, 包括 标签Vocab 和 数据处理类
    tips:
        其中标签Vocab实例化对象必须在api_encode中被调用(add_label)
'''

from torch.utils.data import DataLoader
'''
这里需要导入encode、decode、metric以及dataset的API
'''
from apidataset import api_dataset
from apiencode import api_encode
from apidecode import api_decode
from apimetric import api_metric

class LabelVocabulary:
    UNK = 'UNK'

    def __init__(self):
        self.label2value = {}   #benigh, malignant
        self.value2label = {}

    def _length_(self):
        return len(self.label2value)
    
    def add_label(self, label):
        if label not in self.label2value:
            self.label2value.update({label: len(self.label2value)})   #benigh, malignant : 0/1(相应的value)   
            self.value2label.update({len(self.value2label): label})
    
    def label_to_value(self, label):
        return self.label2value.get(label)
    
    def value_to_label(self, value):
        return self.value2label.get(value) 

class Processor:

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
        dataset_inputs = self.encode(data)
        return api_dataset(*dataset_inputs)
    
    def to_loader(self, data, parameters):
        dataset = self.to_dataset(data)
        return DataLoader(dataset=dataset, **parameters, collate_fn=dataset.collate_fn)