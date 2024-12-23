import os
import sys
sys.path.append('./utils')
import torch
import argparse
from Config import config
import matplotlib.pyplot as plt
from utils.uni_dataprocess import read_tensor, split_dataset
from utils.common import save_model, loss_draw, acc_draw, other_draw
from utils.dataprocess import Uni_processor
from unitrainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action = 'store_true', help = '训练模型')
parser.add_argument('--lr', default = 1e-5, help = '设置学习率', type = float)
parser.add_argument('--weight_decay', default = 1e-3, help = '设置权重衰减', type = float)
parser.add_argument('--epoch', default = 10, help = '设置训练轮数', type = int)
parser.add_argument('--do_test', action = 'store_true', help = '预测测试集数据')
parser.add_argument('--load_model_path', default = None, help = '已经训练好的模型路径', type = str)
args = parser.parse_args()
config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.epoch = args.epoch
config.load_model_path = args.load_model_path


processor = Uni_processor(config)
from model.Unimodal_vision import Univision

model = Univision(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(config, processor, model, device)

def train():
    data = read_tensor(config.labelfile, config.tensor_path)
    train_data, val_data, _ = split_dataset(data, config.train_ratio, config.valid_ratio, config.test_ratio)
    train_loader = processor(train_data, config.train_params)
    val_loader = processor(val_data, config.test_params)
    
    #不需要processor
    best_acc = 0.0
    epoch = config.epoch
    tloss_list, vloss_list = [], []
    acc_list = []
    precision = []
    recall = []
    f1 = []
    Range = range(0, epoch)
    print('这是range的类型', type(Range))
    for e in range(epoch):
        print('-' * 20 + ' ' + 'Epoch ' + str(e+1) + ' ' + '-' * 20)
        tloss, tlosslist = trainer.train(train_loader) #参数是一个Dataloader实例对象，用train函数进行训练，返回训练损失和损失列表
        print('Train Loss: {}'.format(tloss))
        vloss, vacc, report_dict = trainer.valid(val_loader) #valid()函数用于评估模型，并返回验证损失和验证准确率
        print('Valid Loss: {}'.format(vloss))
        print('Valid Acc: {}'.format(vacc))

        tloss_list.append(tloss)
        vloss_list.append(vloss)
        acc_list.append(report_dict['accuracy'])
        precision.append(report_dict['weighted avg']['precision'])
        recall.append(report_dict['weighted avg']['recall'])
        f1.append(report_dict['weighted avg']['f1-score'])
        print('accuracy:{}'.format(report_dict['accuracy']))

        '''
        当每次验证准确率高于最佳准确率时都会更新最佳准确率，并且保存模型
        '''
        if vacc > best_acc:
            best_acc = vacc
            save_model(config.output_path, config.fuse_model_type, model)   #保存训练好的模型
            print('Update best model!')
        print()

        #损失曲线
    loss_draw(tloss_list, vloss_list, Range, os.path.join(config.output_path, 'loss_curve.jpg'))

        #准确率曲线
    acc_draw(acc_list, Range, os.path.join(config.output_path, 'accuracy_curve.jpg'))

        #macro曲线
    other_draw(precision, recall, f1, Range, os.path.join(config.output_path, 'other_curve.jpg'))

def test():
    data = read_tensor(config.labelfile, config.tensor_path)
    _, _, test_data = split_dataset(data)

    test_loader = processor(test_data, config.test_params)

    if config.load_model_path is not None:
        model.load_state_dict(torch.load(config.load_model_path))   #只加载参数字典给model，即最前面的实例对象等价于：model.load_state_dict(torch.load(config.load_model_path),model)
        '''
        我觉得应该是这句:
        '''
        #trainer.model.load_state_dict(torch.load(config.load_model_path))
    
    tloss, tacc, report_dict = trainer.predict(test_loader)
    print('Test Loss: {}'.format(tloss))
    print('Test Acc: {}'.format(tacc))
    print('accuracy:{}'.format(report_dict['accuracy']), '\n', 
          'precision:{}'.format(report_dict['precision']), '\n',
          'recall:{}'.format(report_dict['recall']), '\n', 
          'f1-score:{}'.format(report_dict['f1-score']))

if __name__ == '__main__':
    args = parser.parse_args()
    if args.do_train:
        train()
        
    if args.do_test:
        # if args.load_model_path is None and not args.do_train:
        if config.load_model_path is None and not args.do_train:
            print('请输入已训练好模型的路径load_model_path或者选择添加do_train arg')
        else:
            test()