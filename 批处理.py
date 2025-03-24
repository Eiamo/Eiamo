import numpy as np
nd1=np.random.randn(10000,2,3) # 10000个2*3的矩阵
print(nd1.shape)
# 打乱
np.random.shuffle(nd1)
# 定义一批的大小
size=100
for i in range(0,len(nd1),size):
    x_batch_sum=np.sum(nd1[i:i+size])
    print("第{}次，该批次的数据之和：{}".format(i,x_batch_sum))
