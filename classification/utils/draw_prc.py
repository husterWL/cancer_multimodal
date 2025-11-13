import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.font_manager as fm
fm.fontManager.addfont('/mnt/breast_cancer_multimodal/classification/fonts/TIMES.TTF')

output_path = '/mnt/breast_cancer_multimodal/model'
true1 = np.load(output_path + '/true_labels_unimodal_univision_0723_3.npy')
pre1 = np.load(output_path + '/pred_scores_unimodal_univision_0723_3.npy')
true2 = np.load(output_path + '/true_labels_multimodal_imgemr_0724_1.npy')
pre2 = np.load(output_path + '/pred_scores_multimodal_imgemr_0724_1.npy')
true3 = np.load(output_path + '/true_labels_multimodal_imgkg_0728_7.npy')
pre3 = np.load(output_path + '/pred_scores_multimodal_imgkg_0728_7.npy')

plt.rc('font', family = 'Times New Roman')

if isinstance(true1, torch.Tensor):
    true1 = true1.cpu().numpy()
if isinstance(pre1, torch.Tensor):
    pre1 = pre1.cpu().numpy()
if isinstance(true2, torch.Tensor):
    true2 = true2.cpu().numpy()
if isinstance(pre2, torch.Tensor):
    pre2 = pre2.cpu().numpy()
if isinstance(true3, torch.Tensor):
        true3 = true3.cpu().numpy()
if isinstance(pre3, torch.Tensor):
        pre3 = pre3.cpu().numpy()

# 计算PRC曲线
precision1, recall1, thresholds1 = precision_recall_curve(true1, pre1)
precision2, recall2, thresholds2 = precision_recall_curve(true2, pre2)
precision3, recall3, thresholds3 = precision_recall_curve(true3, pre3)
auprc1 = auc(recall1, precision1)
auprc2 = auc(recall2, precision2)
auprc3 = auc(recall3, precision3)




# 绘制PRC曲线
plt.figure()

plt.plot(recall3, precision3, color = 'darkblue', lw = 1.20, 
    label = f'PRC curve of ImgwithKg (AUPRC = {auprc3:.4f})')
plt.plot(recall2, precision2, color = 'darkgreen', lw = 1.20, 
    label = f'PRC curve of ImgwithEmr (AUPRC = {auprc2:.4f})')
plt.plot(recall1, precision1, color = 'darkorange', lw = 1.20, 
    label = f'PRC curve of Img (AUPRC = {auprc1:.4f})')

# 添加无技能线（随机猜测的性能）
no_skill = len(true1[true1==1]) / len(true1)
plt.axhline(y = no_skill, color = 'r', linestyle = '--', label = 'Random Classifier')


# plt.plot([0, 1], [0, 1], color = 'navy', lw = 1, linestyle = '--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Breast Cancer Multimodal Classifier PRC Curve')
plt.legend(loc = 'lower right')
plt.savefig('prc.svg', dpi = 600, bbox_inches = 'tight', format = 'svg')

display = PrecisionRecallDisplay(precision = precision1, recall = recall1)
display.plot(name=f'AUPRC={auprc1:.6f}')