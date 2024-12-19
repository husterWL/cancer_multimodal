import torch
import argparse
from Config import config
import matplotlib.pyplot as plt
from utils.uni_dataprocess import read_tensor, split_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true', help='训练模型')

processor = Processor(config)
from model.Unimodal_vision import VisionModel

model = VisionModel(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(config, processor, model, device)

def train():
    data = read_tensor(config.labelfile, config.tensor_path)
    train_data, val_data = split_dataset(data)
    
    #不需要processor
    best_acc = 0.0
    epoch = config.epoch
    tloss_list, vloss_list = [], []
    acc_list = []
    precision = []
    recall = []
    f1 = []
    Range = range(0, epoch)
    for e in range(epoch):
        print('-' * 20 + ' ' + 'Epoch ' + str(e+1) + ' ' + '-' * 20)
        

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.do_train:
        print('Do nothing!')
    else:
        # 读取数据
        train_iter = torch.utils.data.DataLoader(dataset=config.TRAIN_DATASET, batch_size=config.BATCH_SIZE, shuffle=True)
        test_iter = torch.utils.data.DataLoader(dataset=config.TEST_DATASET, batch_size=config.BATCH_SIZE, shuffle=False)

        # 训练模型
        for epoch in range(10):
            for i, (features, labels) in enumerate(train_iter):
                print(features, labels)
                break
            break

        # 测试模型
        for features, labels in test_iter:
            print(features, labels)
            break

        # 保存模型
        torch.save(model.state_dict(), config.MODEL_PATH)

        # 绘制损失曲线
        plt.plot(losses)