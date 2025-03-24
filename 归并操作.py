import torch
a=torch.linspace(0,10,6)
print(a)
a=a.view((2,3))
print(a)
# 沿着y轴累加
b=a.sum(dim=0)
print(b.shape)
# 保留含1的维度
b=a.sum(dim=0,keepdim=True)
print(b.shape)