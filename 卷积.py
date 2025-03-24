import torch

def cust_conv2d(X, K):
    """ 实现卷积运算 """
    # 获取卷积核形状
    h, w = K.shape
    # 初始化输出值 Y
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 实现卷积运算
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[1.0,1.0,1.0,0.0,0.0], [0.0,1.0,1.0,1.0,0.0],
                  [0.0,0.0,1.0,1.0,1.0],[0.0,0.0,1.0,1.0,0.0],[0.0,1.0,1.0,0.0,0.0]])
K = torch.tensor([[1.0, 0.0,1.0], [0.0, 1.0,0.0],[1.0, 0.0,1.0]])
cust_conv2d(X, K)
print(cust_conv2d(X, K))