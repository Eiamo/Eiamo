import numpy as np
from matplotlib import  pyplot as  plt
from sympy.printing.pretty.pretty_symbology import line_width

np.random.seed(100)
x=np.linspace(-1,1,100).reshape(100,1)
y=3*np.power(x,2)+2+0.2*np.random.rand(x.size).reshape(100,1)
print(y)
plt.scatter(x,y)
plt.show()
# 初始化参数
w1=np.random.rand(1,1)
b1=np.random.rand(1,1)
# 训练模型
r1=0.001
for i in range(800):
    y_pred=np.power(x,2)*w1+b1

    loss=0.5*(y_pred)**2
    loss=loss.sum()

    grad_w=np.sum((y_pred-y)*np.power(x,2))
    grad_b=np.sum((y_pred-y))

    w1-=r1*grad_w
    b1-=r1*grad_b

plt.plot(x,y_pred,'r-',label='predict',linewidth=4)
plt.scatter(x,y,color='blue',marker='o',label='true')
plt.xlim(-1,1)
plt.ylim(2,6)
plt.legend()
plt.show()
print(w1,b1)