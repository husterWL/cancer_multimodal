'''
encode api: 将原始data数据转化成APIDataset所需要的数据
    tips:
        ! 必须调用labelvocab的add_label接口将标签加入labelvocab字典
'''

from tqdm import tqdm
# from transformers import AutoTokenizer  #不需要bert
from torchvision import transforms


def api_encode(data, labelvocab, config):
    
    labelvocab.add_label('benign')
    labelvocab.add_label('malignant')

    '''对读取的data进行预处理'''

    guids, encoded_EHRs, encoded_imgs, encoded_KGs, encoded_labels = [], [], [], [], []
    for line in tqdm(data, desc='----- [Encoding]'):
        guid, img, emr, kg, label = line['id'], line['tensor'], line['emr'], line['kg'], line['label']    #line若为字典，不能直接解包赋予变量；若为元组或列表，可以直接解包赋予变量


        guids.append(guid)
        encoded_imgs.append(img)
        encoded_EHRs.append(emr)
        encoded_KGs.append(kg)
        encoded_labels.append(labelvocab.label_to_value(label))
    
    return guids, encoded_imgs, encoded_EHRs, encoded_KGs, encoded_labels
