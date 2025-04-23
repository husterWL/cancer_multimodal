'''
encode api: 将原始data数据转化成APIDataset所需要的数据
    tips:
        ! 必须调用labelvocab的add_label接口将标签加入labelvocab字典
'''

from tqdm import tqdm
# from transformers import AutoTokenizer  #不需要bert
from torchvision import transforms


def api_encode(data, labelvocab, config):
    
    labelvocab.add_label('benigh')
    labelvocab.add_label('maglinant')

    '''EHR处理'''


    '''image处理'''
    img_transform = transforms.Compose([    #构建一个图像处理器，进行图像预处理操作，对图片进行归一化、大小缩放等等

    ])


    '''对读取的data进行预处理'''

    guids, encoded_EHRs, encoded_imgs, encoded_labels = [], [], [], []
    for line in tqdm(data, desc='----- [Encoding]'):
        guid, ehr, img, label = line
        guids.append(guid)
        encoded_EHRs.append(ehr)
        encoded_imgs.append(img_transform(img))
        encoded_labels.append(labelvocab.label_to_id(label))
    
    return(guids, encoded_EHRs, encoded_imgs, encoded_labels)
