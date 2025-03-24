import numpy as np
from numpy import random as nr

a=np.arange(1,25)
c1=nr.choice(a,size=(3,4))
c2=nr.choice(a,size=(3,4),replace=False)
# 下中p指定每个元素对应的抽取概率，默认概率相同
c3=nr.choice(a,size=(3,4),p=a / np.sum(a))
print(f'原始数据a{a}')
print(f'随机可重复抽取{c1}')
print(f'随机不重复抽取{c2}')
print(f'随机按概率抽取{c3}')