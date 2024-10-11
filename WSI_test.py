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
#读取
# OPENSLIDE_PATH = 'D:/Anaconda_WL/openslide-bin-4.0.0.6-windows-x64/bin'

# import os
# if hasattr(os, 'add_dll_directory'):
#     # Windows
#     with os.add_dll_directory(OPENSLIDE_PATH):
#         import openslide
# else:
#     import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import openslide

