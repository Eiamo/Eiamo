import numpy as np
a=np.arange(0,40,10).reshape(4,1)
b=np.arange(0,3)
print("a的形状：{},b的形状：{}".format(a.shape,b.shape))# format是格式化
c=a+b
print("c的形状：{}".format(c.shape))
print(c)