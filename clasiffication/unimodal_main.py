import torch
import argparse
from Config import config
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true', help='训练模型')

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