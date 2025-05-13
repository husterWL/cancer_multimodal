import os
import sys
# 指定你想要的根目录
# os.chdir('/path/to/your/directory')

# 打印当前工作目录以验证
print(os.getcwd())
print(os.path.abspath('....'))

with open('./classification/data/train_id.txt', 'r') as f:
    for line in f.readlines():
        print(line.strip('\n'))

with open('./classification/data/train_id.txt', 'r') as f:
    for line in f.readlines():
        print('1')