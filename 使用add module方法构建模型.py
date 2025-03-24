import torch
import torch.nn as nn

# 定义超参数
in_dim = 28 * 28  # 输入维度，例如28*28的图像展开后的维度
n_hidden_1 = 300  # 第一个隐藏层神经元数量
n_hidden_2 = 100  # 第二个隐藏层神经元数量
out_dim = 10  # 输出维度，比如分类任务中的类别数

# 构建模型
Seq_module = nn.Sequential()
Seq_module.add_module("flatten", nn.Flatten())
Seq_module.add_module("linear1", nn.Linear(in_dim, n_hidden_1))
Seq_module.add_module("bn1", nn.BatchNorm1d(n_hidden_1))
Seq_module.add_module("relu1", nn.ReLU())
Seq_module.add_module("linear2", nn.Linear(n_hidden_1, n_hidden_2))
Seq_module.add_module("bn2", nn.BatchNorm1d(n_hidden_2))
Seq_module.add_module("relu2", nn.ReLU())
Seq_module.add_module("out", nn.Linear(n_hidden_2, out_dim))
Seq_module.add_module("softmax", nn.Softmax(dim=1))

# 查看模型
print(Seq_module)

