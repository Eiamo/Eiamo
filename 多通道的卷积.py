import torch


def cust_conv2d(X, K):
    """ 实现二维卷积运算 """
    # 获取卷积核形状
    h, w = K.shape
    # 初始化输出值 Y
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 实现卷积运算
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def corr2d_mutl_in(X, K):
    h, w = K.shape[1], K.shape[2]
    value = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for x, k in zip(X, K):
        value = value + cust_conv2d(x, k)
    return value


# 修改后的输入数据定义，确保每个通道内矩阵的行元素数量一致
X = torch.tensor([[[1., 0., 1., 0., 2.], [1., 3., 2., 1., 0], [1., 1., 0., 1., 0]],
                  [[2., 3., 2., 1., 3.], [0., 2., 0., 1., 0.]],
                  [[3., 4., 5., 6., 7.], [8., 9., 10., 11., 12.], [13., 14., 15., 16., 17.]]])
K = torch.tensor([[[0., 0., 1., 0., 0.], [0., 0., 0., 0., 2.], [0., 0., 1., 0., 0.]],
                  [[2., 0., 1., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 3., 0., 0.]],
                  [[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [0., 0., 0., 0., 2.]]])

result = corr2d_mutl_in(X, K)
print(result)