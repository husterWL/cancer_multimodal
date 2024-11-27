'''
视觉模型、映射结构、语言模型、结构化模型
PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers——框架图
Richer fusion network for breast cancer classification based on multimodal data——WSI+结构化数据
'''
'''
2024年10月7日
需要融合image data与structured EMR data
重点参考richer fusion network

首先解决WSI处理问题，如何extract特征
'''


'''
具体可参考clam仓库
https://github.com/mahmoodlab/CLAM
'''

#读取

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import openslide

slidelist = os.listdir("D:/BaiduNetdiskDownload/mutimodal_breast_cancer/Benign_pathological_image1")
# "C:/Users/WL/Desktop/benign_S0000004_1.tiff"

# slidelist = os.listdir("C:/Users/WL/Desktop/image")
for slidei in slidelist[:]:
    print(slidei)
    idi = slidei.split('_')[1] + '_' + slidei.split('_')[2]
    # print(idi)
    
    
    # open slide.svs
    # slide_0 = openslide.OpenSlide('C:/Users/WL/Desktop/image/{}'.format(slidei))
    slide_0 = openslide.open_slide('D:/BaiduNetdiskDownload/mutimodal_breast_cancer/Benign_pathological_image1/{}'.format(slidei))
    levels = slide_0.level_dimensions   #((2048, 1536),)
    print("stored slide size pyramid",levels)

    # fetch levels[2] size of whole WSI region 
    # slide = slide_0.read_region((0, 0), 1, levels[0])
    # slide = np.asarray(slide)
    # print("fetched shape", slide.shape)

    # origin slide is in RGBA format, convert it to RGB and save to model data dir
    # slide = cv2.cvtColor(slide, cv2.COLOR_RGBA2RGB)
    # skio.imsave('../data/WSI/slide_{}.png'.format(idi),slide.astype("uint8"))
    # plt.imshow(slide)
    # plt.show()
'''
openslide.ImageSlide、openslide.open_slide、openslide.OpenSlide不一样
'''


