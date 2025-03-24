import torch
from torch import nn

in_dim, n_hidden_1, n_hidden_2, out_dim=28 * 28, 300, 100, 10

Seq_arg = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_dim,n_hidden_1),
    nn.BatchNorm1d(n_hidden_1),
    nn.ReLU(),
    nn.Linear(n_hidden_1, n_hidden_2),
    nn.BatchNorm1d(n_hidden_2),
    nn.ReLU(),
    nn.Linear(n_hidden_2, out_dim),
    nn.Softmax(dim=1)
)

print(Seq_arg)