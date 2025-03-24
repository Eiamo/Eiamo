import torch
from torch import  nn
import  torch.nn.functional as  f

class Model_seq(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):# 定义类的函数方法，传入后面的数据初始化
        super(Model_seq,self).__init__()# 调用nn。model的构造函数
        self.flatten=nn.Flatten()            # 创建一个展平层
        self.linear1=nn.Linear(in_dim,n_hidden_1)# 第一层
        self.bn1=nn.BatchNorm1d(n_hidden_1)# 把第一层的数据分批
        self.linear2= nn.Linear( n_hidden_1,n_hidden_2)# 把第一层的数据穿到第二层
        self.bn2 = nn.BatchNorm1d(n_hidden_2)# 第二层分批处理
        self.out=nn.Linear(n_hidden_2,out_dim)# 输出第二层的数据和维度

    def forward(self, x):
        x = self.flatten(x)# 展平数据
        x = self.linear1(x)# 把x传到第一层
        x = self.bn1(x)# 批归一化处理
        x = f.relu(x)# 激活函数引入
        x = self.linear2(x)# 到第二层
        x = self.bn2(x) #批归一化
        x = f.relu(x) # 激活函数
        x = self.out(x)
        x = f.softmax(x, dim=1)
        return x
in_dim, n_hidden_1, n_hidden_2, out_dim=28 * 28, 300, 100, 10
model_seq = Model_seq(in_dim, n_hidden_1, n_hidden_2, out_dim)
print(model_seq)
