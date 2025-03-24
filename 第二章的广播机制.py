import torch
import numpy as np
a=np.arange(0,40,10).reshape(4,1)
b=np.arange(0,3)

a1=torch.from_numpy(a)
b1=torch.from_numpy(b)
c=a1+b1
print(c)
# 自动广播 不用手动配置