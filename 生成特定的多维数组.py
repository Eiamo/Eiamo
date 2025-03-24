import numpy as np
from Demos.mmapfile_demo import fname
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

# 生成全是0的3*3
nd1=np.zeros([3,3])
print(nd1)
# 和nd1一样的全0阵
nd2=np.zeros_like(nd1)
print(nd2)
# 全是1的3*3
nd3=np.ones([3,3])
print(nd3)
# 把生成的数据暂时保存起来
nd4=np.random.random([3,3])
np.savetxt(X=nd4,fname='dataset/hymenoptera_data/val/nd9')
nd5=np.loadtxt('dataset/hymenoptera_data/val/nd9')
print(nd5)