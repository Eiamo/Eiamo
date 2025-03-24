import torch
import torch.nn as nn

# 1）定义输入和输出
X = torch.tensor([[10.,10.,10.,0.,0.,0.], [10.,10.,10.,0.,0.,0.], [10.,10.,10.,0.,0.,0.],
                  [10.,10.,10.,0.,0.,0.], [10.,10.,10.,0.,0.,0.], [10.,10.,10.,0.,0.,0.]])
X = X.reshape((1, 1, 6, 6))
Y = torch.tensor([[0., 30.,30.,0.], [0., 30.,30.,0.], [0., 30.,30.,0.], [0., 30.,30.,0.]])
Y = Y.reshape((1, 1, 4, 4))

# 2）训练卷积层
# 构造一个二维卷积层，它具有1个输出通道和形状为(3, 3)的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 3), bias=False)
lr = 0.001  # 学习率
# 定义损失函数
loss_fn = torch.nn.MSELoss()
for i in range(400):
    Y_pre = conv2d(X)
    loss = loss_fn(Y_pre, Y)
    conv2d.zero_grad()
    loss.backward()
    # 更新卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad# 切片
    if (i + 1) % 100 == 0:
        print(f'epoch {i+1}, loss {loss.sum():.4f}')# 对损失函数求和且保留4位小数

# 3）查看卷积核
conv2d.weight.data = conv2d.weight.data.reshape((3, 3))
print("运行结果如下：")
print(conv2d.weight.data)