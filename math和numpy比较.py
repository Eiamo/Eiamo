import time
import math
import numpy as np
x=[i*0.001 for i in np.arange(1000000)]
start=time.perf_counter()
for i,t in enumerate(x):
    x[i]=math.sin(t)
print("math.sin:是",time.perf_counter() - start)

x=[i*0.001 for i in np.arange(1000000)]
x=np.array(x)
start=time.perf_counter()
np.sin(x)
print("np。sin是：",time.perf_counter() - start)