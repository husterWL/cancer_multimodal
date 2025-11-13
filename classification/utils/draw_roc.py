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

# 计算ROC曲线   
fpr1, tpr1, thresholds1 = roc_curve(true1, pre1, drop_intermediate = False)
roc_auc1 = auc(fpr1, tpr1)
fpr2, tpr2, thresholds2 = roc_curve(true2, pre2, drop_intermediate = False)
roc_auc2 = auc(fpr2, tpr2)
fpr3, tpr3, thresholds3 = roc_curve(true3, pre3, drop_intermediate = False)
roc_auc3 = auc(fpr3, tpr3)

# 计算PRC曲线

# 绘制ROC曲线
plt.figure()
plt.plot(fpr3, tpr3, color = 'darkblue', lw = 1.20, 
    label = f'ROC curve of ImgwithKg (AUC = {roc_auc3:.4f})')
plt.plot(fpr2, tpr2, color = 'darkgreen', lw = 1.20, 
    label = f'ROC curve of ImgwithEmr (AUC = {roc_auc2:.4f})')
plt.plot(fpr1, tpr1, color = 'darkorange', lw = 1.20, 
    label = f'ROC curve of Img (AUC = {roc_auc1:.4f})')


plt.plot([0, 1], [0, 1], color = 'navy', lw = 1, linestyle = '--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Breast Cancer Multimodal Classifier ROC Curve')
plt.legend(loc = 'lower right')
plt.savefig('roc.svg', dpi = 600, bbox_inches = 'tight', format = 'svg')