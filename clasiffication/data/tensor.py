import torch

tensor = torch.rand(5, 3)
print(tensor)

tensor1 = tensor.new_ones(5, 3, dtype=torch.double)      # new_* 方法创建的新张量与原张量具有相同的大小和类型
tensor1 = tensor.new_ones(5, 3, dtype=torch.double, device='cuda:0')   # 也可以指定设备
print(tensor1)
tensor2 = tensor.view(5, 3)                               # view 方法创建的张量与原张量共享底层存储
print(tensor2)

#张量的保存和加载
torch.save(tensor, 'tensor.pt')
# 或者，保存一个字典中的多个张量
torch.save({'tensor1': tensor1, 'tensor2': tensor2}, 'tensors.pt')

# 加载整个字典
tensors = torch.load('tensors.pt')
tensor1 = tensors['tensor1']
tensor2 = tensors['tensor2']

# 加载单个张量
tensor = torch.load('tensor.pt')

tensor_cancer = torch.load(
    r"D:\BaiduNetdiskDownload\multimodal_breast_cancer\Features_directory\pt_files\benign_S0000004_1.pt")

print(tensor_cancer.shape)  # 输出：torch.Size([patchs_num, 1048])
print(tensor_cancer[1])