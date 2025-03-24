import numpy as np
np.random.seed(2019)
nd11=np.random.random([10])
# 获取第四个元素
nd11[3]
# 截取一段数据
nd1=nd11[3:6]
# 截取固定间隔数据
nd11[1:6:2]
# 倒序取数
nd11[::-2]
# 截取某个区域
nd12=np.arange(25).reshape([5,5])
nd12[1:3,1:3]
# 截取在某个之间的数
nd13=nd12[(nd12>3)&(nd12<10)]
# 截取2.3行
nd14=nd12[[1,2]]
# 截取2.3列
nd2=nd12[:,1:3]
print(f'nd11是:{nd11}')
print(f'nd12是:{nd12}')
print(f'nd13是:{nd13}')
print(f'nd14是:{nd14}')
print(f'nd1是:{nd1}')
print(f'nd2是:{nd2}')