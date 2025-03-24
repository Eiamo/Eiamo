import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

input_size = 1
output_size = 1
num_epoches = 100
learning_rate = 0.01

dtype = torch.FloatTensor
writer = SummaryWriter(log_dir = 'logs', comment='Linear')

np.random.seed(100)
x_train = np.linspace(-1, 1, 100).reshape(100,1)
y_train = 3*np.power(x_train, 2) +2 + 0.2*np.random.randn(x_train.size).reshape(100,1)

model = nn.Linear(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    inputs = torch.from_numpy(x_train).type(dtype)
    targets = torch.from_numpy(y_train).type(dtype)

    output = model(inputs)
    loss = criterion(output, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 保存 loss 与 epoch 数值
    writer.add_scalar('训练损失值', loss, epoch)