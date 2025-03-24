# 格式arange（[start,]stop[,step,],dtype=none）
import numpy as np


print(np.arange(10))
nd1=np.arange(1,4,0.5)
print(nd1,type(nd1))
# linspace根据范围以及等分数生成一个线性等分量
nd2=np.linspace(0,1,10)
print(nd2)
