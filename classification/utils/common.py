import os
import json
import chardet
import torch
from tqdm import tqdm   #进度条库
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import roc_curve, auc

def split_dataset(path, train_ratio, valid_ratio, test_ratio):   #分割数据集
    
    data = os.listdir(path)
    assert train_ratio + valid_ratio + test_ratio == 1, 'Ratio error.'   #判断划分是否正确
    
    train_nums = int(len(data) * train_ratio)
    valid_nums = int(len(data) * valid_ratio)
    test_nums = int(len(data) * test_ratio)
    
    train, valid, test = [], [], []
    for root, dirs, files in os.walk(path):
        random.shuffle(files)   #打乱文件顺序
        for file in files:
            if len(train) < train_nums:
                train.append(os.path.join(root, file))
            elif len(valid) < valid_nums:
                valid.append(os.path.join(root, file))
            else:
                test.append(os.path.join(root, file))
    return train, valid, test

def data_format(input_path, data_dir, output_path):
    data = []

def read_from_file():
    pass

def write_to_file(path, outputs):   #可以用来输出测试结果
    with open(path, 'w') as f:
        for line in tqdm(outputs, desc='----- [Writing]'):
            f.write(line)
            f.write('\n')
        f.close()

def save_model(output_path, model_type, model):
    '''可以用来输出训练后的模型'''
    output_model_dir = os.path.join(output_path, model_type)    #输出模型的保存目录
    if not os.path.exists(output_model_dir): os.makedirs(output_model_dir)
    model_to_save = model.module if hasattr(model, 'module') else model     # Only save the model it-self
    output_model_file = os.path.join(output_model_dir, 'pytorch_model_multimodal_bicrossmodel_0702_1.bin')
    torch.save(model_to_save.state_dict(), output_model_file)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))

def loss_draw(tloss_list, vloss_list, x, dirc):
    plt.figure(figsize = (32, 24))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    plt.plot(x, tloss_list, 'k-', label = 'train_loss', linewidth = 2.0)
    plt.plot(x, vloss_list, 'r--', label ='vloss_list', linewidth = 2.0)
    plt.ylabel('loss', fontsize = 40)
    plt.xlabel('epoch', fontsize = 40)
    plt.rc('legend', fontsize = 40)
    plt.xticks(fontsize = 40)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize = 40)
    plt.legend(['train', 'valid'], loc = 'upper right')
    plt.savefig(dirc)

def acc_draw(acc_list, x, dirc):
    plt.figure(figsize = (32, 24))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    plt.plot(x, acc_list, 'k-', label = 'accuracy', linewidth = 2.0)
    plt.ylabel('accuracy', fontsize = 40)
    plt.xlabel('epoch', fontsize = 40)
    plt.xticks(fontsize = 40)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize = 40)
    plt.rc('legend', fontsize = 40)
    plt.legend()
    plt.savefig(dirc)

def other_draw(p, r, f1, x, dirc):
    plt.figure(figsize = (32, 24))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    plt.plot(x, p, 'k', linestyle = 'solid', label = 'precision', linewidth = 2.0)
    plt.plot(x, r, 'k', linestyle = 'dotted', label = 'recall', linewidth = 2.0)
    plt.plot(x, f1, 'k', linestyle = 'dashed', label = 'f1', linewidth = 2.0)
    plt.xlabel('epoch', fontsize = 40)
    plt.xticks(fontsize = 40)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize = 40)
    plt.rc('legend', fontsize = 40)
    plt.legend(['precesion', 'recall', 'F1'], loc = 'upper right')
    plt.savefig(dirc)

def earlystop_draw(tloss_list, vloss_list, dirc):
    pass
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(32,24))
    plt.plot(range(1,len(tloss_list)+1),tloss_list, label='Training Loss')
    plt.plot(range(1,len(vloss_list)+1),vloss_list,label='Validation Loss')

    # find position of lowest validation loss
    minposs = vloss_list.index(min(vloss_list))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(tloss_list)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(dirc, bbox_inches='tight')
    # plt.savefig(dirc, bbox_inches = 'tight')

def roc_draw(true_labels, pred_scores, dirc):

    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.cpu().numpy()

    # 计算ROC曲线   
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)

    dense_thresholds = np.linspace(0, 1, 1000)  # 1000个阈值点
    tpr_dense = []
    fpr_dense = []
    true_tensor = torch.tensor(true_labels)
    pred_tensor = torch.tensor(pred_scores)
    for thresh in dense_thresholds:
        preds = (pred_tensor >= thresh).float()
        tp = ((preds == 1) & (true_tensor == 1)).sum().item()
        fp = ((preds == 1) & (true_tensor == 0)).sum().item()
        tn = ((preds == 0) & (true_tensor == 0)).sum().item()
        fn = ((preds == 0) & (true_tensor == 1)).sum().item()
        
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_dense.append(tpr_val)
        fpr_dense.append(fpr_val)
    dense_auc = auc(fpr_dense, tpr_dense)
    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot(fpr_dense, tpr_dense, color='red', lw=1.5, 
             label=f'密集曲线 (AUC = {dense_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multimodal Classifier ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('multimodal_roc.png')
    # plt.savefig(dirc)