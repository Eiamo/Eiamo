import numpy as np
X=np.random.rand(2,3)
def sigmoid(X):
    return 1/(1+np.exp(-X))
def relu(X):
    return  np.maximum(0,X)
def softmax(X):
    return np.exp(X)/np.sum(np.exp(X))


print('参数x的形状：',X.shape)
print('激活函数sigmoid输出形状',sigmoid(X).shape)
print('激活函数relu输出形状',relu(X).shape)
print('激活函数softmax输出形状',softmax(X).shape)