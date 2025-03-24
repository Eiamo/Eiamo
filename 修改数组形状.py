import  numpy as np
from numpy.ma.core import shape

nd1=np.arange(10)
print(nd1)
# 将nd1变成2行5列，不变本身
print(nd1.reshape(2,5))
# -1来表示其他
print(nd1.reshape(5,-1))
# 改变本身resize

nd1.resize(2,5)
print(nd1)
# 转置函数T
nd2=np.arange(12).reshape(3,4)
print(nd2)
print(nd2.T)
# ravel展开成1维
nd3=np.arange(6).reshape(2,-1)
print(f'原本的：{nd3}')
nd4=nd3.ravel('F')
print(f'按照列优先，展平的：{nd4}')
nd5=nd3.ravel()
print(f'按照行优先，展平的：{nd5}')

# 展开成1维 ，且会产生副本
a=np.floor(10*np.random.random((3,4)))
print(f'原本的a：{a}')
print('使用flatten后的：')
print(a.flatten(order='c'))

# squeeze降维且去掉1
b=np.arange(3).reshape(3,1)
print('原本的b是')
print(b.shape)
print('变化的b')
print(b.squeeze().shape)

# transpose用于高维矩阵的轴对换
print("使用transpose")
c=np.arange(24).reshape(2,3,4)
print('原本的c')
print(c)
print('后来c')
print(c.shape)
print('最后的c')
print(c.transpose(1,2,0).shape)



nd10=np.arange(24).reshape(2,3,4)
print(nd10.shape)
print(nd10.transpose(1,2,0).shape)