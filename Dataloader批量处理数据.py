import torch
from torch.utils import data
import numpy as np


# 自定义数据集类
class MyDataset(data.Dataset):
    def __init__(self):
        # 这里简单定义一些数据和标签示例，实际应用中替换为真实数据
        self.data = np.array([[1, 2], [3, 4], [5, 6]])
        self.labels = np.array([0, 1, 2])

    def __getitem__(self, index):
        # 将numpy数组转换为torch的Tensor类型
        input_data = torch.from_numpy(self.data[index])
        label = torch.tensor(self.labels[index])
        return input_data, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # 创建数据集实例
    dataset = MyDataset()

    # 使用DataLoader加载数据
    data_loader = data.DataLoader(
        dataset,
        batch_size=2,  # 这里设置为2，每次加载2个样本，可以根据需求调整
        shuffle=True,  # 训练时通常打乱数据增加随机性
        sampler=None,
        batch_sampler=None,
        num_workers=0,  # 如果在Windows上有问题，先设置为0
        collate_fn=data.default_collate,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None
    )

    # 遍历数据加载器
    for batch_data, batch_labels in data_loader:
        print("Batch data shape:", batch_data.shape)
        print("Batch labels:", batch_labels)