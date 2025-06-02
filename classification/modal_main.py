import os
import sys
sys.path.append('./utils')
# print(os.getcwd())
import torch
import argparse
from Config import config
from utils.data_read import read_tensor, split_dataset, read_tensor_emr
from utils.common import save_model, loss_draw, acc_draw, other_draw, earlystop_draw
from utils.dataprocess import Uni_processor, Processor
from unitrainer import Trainer
from trainer import multitrainer
from early_stopping_pytorch import EarlyStopping



parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action = 'store_true', help = '训练模型')
parser.add_argument('--lr', default = 1e-4, help = '设置学习率', type = float)
parser.add_argument('--weight_decay', default = 1e-4, help = '设置权重衰减', type = float)
parser.add_argument('--epoch', default = 10, help = '设置训练轮数', type = int)
parser.add_argument('--do_test', action = 'store_true', help = '预测测试集数据')
parser.add_argument('--load_model_path', default = None, help = '已经训练好的模型路径', type = str)
parser.add_argument('--model_type', default = 'multimodal', action = 'store', help = '是否多模态融合', type = str)
parser.add_argument('--fusion_type', default = 'KGBased', action = 'store', help = '多模态融合方式', type = str)

args = parser.parse_args()
config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.epoch = args.epoch
# config.load_model_path = args.load_model_path

config.model_type = args.model_type
config.fusion_type = args.fusion_type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config.model_type == 'unimodal':
    processor = Uni_processor(config)
    if config.fusion_type == 'Univision':
        from model.Unimodal import Univision
        model = Univision(config)
    elif config.fusion_type == 'Univision_sa':
        from model.Unimodal import Univision_sa
        model = Univision_sa(config)
    trainer = Trainer(config, processor, model, device)
elif config.model_type == 'multimodal':
    processor = Processor(config)
    if config.fusion_type == 'Concatenate':
        from model.Multimodal import Concatmodel as FuseModel
    if config.fusion_type == 'Bicrossmodel':
        from model.Multimodal import Bicrossmodel as FuseModel
    if config.fusion_type == 'KGBased':
        from model.Multimodal import KGBased as FuseModel
    model = FuseModel(config)
    trainer = multitrainer(config, processor, model, device)



def train():

    if not config.model_type == 'unimodal':
        data = read_tensor_emr(config.labelfile, config.tensor_path, config.emr_path)
    
    else:
        data = read_tensor(config.labelfile, config.tensor_path)
        
    train_data = []
    val_data = []
    '''
    这里由于data是字典的列表，所以按照id.txt文件进行划分需要遍历，单纯的遍历会产生大量的IO时间，需要进行优化
    在这里重新将以字典为元素的列表再映射为一个字典，当然这只适合id唯一的情况
    '''
    lookup_data = {dic['id']: dic for dic in data}
    with open('./data/exclusion_train_id.txt', 'r') as f:
        for line in f.readlines():
            train_data.append(lookup_data[line.strip('\n')])
    with open('./data/exclusion_valid_id.txt', 'r') as f:
        for line in f.readlines():
            val_data.append(lookup_data[line.strip('\n')])

    # else:
    #     data = read_tensor(config.labelfile, config.tensor_path)
        # train_data, val_data, _ = split_dataset(data, config.train_ratio, config.valid_ratio, config.test_ratio)
    

    
    train_loader = processor(train_data, config.train_params)
    valid_loader = processor(val_data, config.test_params)
    
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

    early_stop = EarlyStopping(patience = config.patience, verbose = True, path = config.output_path + '\\checkpoint.pt')
    
    for e in range(epoch):
        print('-' * 20 + ' ' + 'Epoch ' + str(e+1) + ' ' + '-' * 20)
        tloss, tlosslist = trainer.train(train_loader) #参数是一个Dataloader实例对象，用train函数进行训练，返回训练损失和损失列表
        print('Train Loss: {}'.format(tloss))
        vloss, vacc, report_dict = trainer.valid(valid_loader) #valid()函数用于评估模型，并返回验证损失和验证准确率
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
            save_model(config.output_path, config.model_type, model)
            print('Update best model!')
        print()

        early_stop(vloss, trainer.model)
  
        if early_stop.early_stop:
            print("Early stopping")
            last_epoch = e + 1
            print(last_epoch)
            break

    #损失曲线
    loss_draw(tloss_list, vloss_list, range(0, last_epoch), os.path.join(config.output_path, 'loss_curve.jpg'))
    #准确率曲线
    acc_draw(acc_list, range(0, last_epoch), os.path.join(config.output_path, 'accuracy_curve.jpg'))
    #macro曲线
    other_draw(precision, recall, f1, range(0, last_epoch), os.path.join(config.output_path, 'other_curve.jpg'))
    #early_stop图
    earlystop_draw(tloss_list, vloss_list, os.path.join(config.output_path, 'early_stop.jpg'))


def test():

    if not config.model_type == 'unimodal':
        data = read_tensor_emr(config.labelfile, config.tensor_path, config.emr_path)
    
    else:
        data = read_tensor(config.labelfile, config.tensor_path)
    test_data = []
    lookup_data = {dic['id']: dic for dic in data}
    with open('./data/exclusion_test_id.txt', 'r') as f:
        for line in f.readlines():
            test_data.append(lookup_data[line.strip('\n')])

    test_loader = processor(test_data, config.test_params)

    if config.load_model_path is not None:
        model.load_state_dict(torch.load(config.load_model_path))   #只加载参数字典给model，即最前面的实例对象等价于：model.load_state_dict(torch.load(config.load_model_path),model)
        '''
        我觉得应该是这句:
        '''
        #trainer.model.load_state_dict(torch.load(config.load_model_path))
    
    tacc, report_dict = trainer.predict(test_loader)
    print('Test Acc: {}'.format(tacc))
    print(
          'accuracy:{}\n'.format(report_dict['accuracy']), 
          'precision:{}\n'.format(report_dict['weighted avg']['precision']),
          'recall:{}\n'.format(report_dict['weighted avg']['recall']),
          'f1-score:{}'.format(report_dict['weighted avg']['f1-score'])
        )
    


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