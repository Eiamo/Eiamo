import torch
from torch.utils.tensorboard import SummaryWriter

# 创建SummaryWriter对象，日志保存在'runs/simple_example'目录下
writer = SummaryWriter('logs666')

# 模拟10个步骤，记录每个步骤的数值
for step in range(10):
    value = step ** 2  # 这里简单用步骤数的平方作为示例数值
    writer.add_scalar('Simple Value', value, step)

# 关闭SummaryWriter
writer.close()