import time
import numpy as np
x1=np.random.rand(1000000)
x2=np.random.rand(1000000)
# 使用np计算向量点积
tic=time.process_time()
dot=np.dot(x1,x2)
toc=time.process_time()
print("dot是："+str(dot)+"电脑时间："+str(100000*(toc-tic)))