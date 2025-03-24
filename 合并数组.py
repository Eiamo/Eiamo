import  numpy as np
from sympy.codegen import Print

# 1维合并数组
# a=np.array([1,2,3])
# b=np.array([4,5,10])
# print(np.append(a,b))

# 合并多维
a=np.arange(4).reshape(2,2)
b=np.arange(4).reshape(2,2)
# 行合并
print('按照行合并')
c=np.append(a,b,axis=0)
print(c)
print('按照行合并后的数据维度',c.shape)
d=np.append(a,b,axis=1)
print('按照列合并')
print(d)
print('按照列合并后的数据维度',d.shape)


# 沿着指定轴链接数组或者矩阵
nd1=np.array([[1,2],[3,4]])
nd2=np.array([[5,6]])

nd3=np.concatenate((nd1,nd2),axis=0)
print('nd3是：')
print(nd3)
nd4=np.concatenate((nd1,nd2.T),axis=1)
print('nd4是：')
print(nd4)


# 沿着指定轴堆叠
nd5=np.array([[5,6],[7,8]])
print('nd5和nd1按照行堆叠是：')
print(np.stack((nd1,nd5),axis=1))


# zip
print('zip项目')
e=np.array([[1,2],[3,4]])

f=np.array([[5,6],[7,8]])
g=zip(e,f)
for i,j in g:
    print(i)
    print(j)


print('使用zip组合两个向量')
a1=[1,2,3]
b1=[2,3,4]
c1=zip(a1,b1)
for i,j in c1:
    print(i,end=",")
    print(j)