# 生成（4.4），值在0和1之间的随机数
import numpy as np
print("生成形状4×4的，0-1的随机数")
print(np.random.random((4,4)),'\n')# 尾部的\n是换行
# 生成一个[1，50）之间的数组，为3×3
# 起始值low=0，终止值high=1
print('生成3×3的，在low和high内的随机整数')
print(np.random.randint(low=1,high=50,size=(3,3)))
# 满足正太分布
print(np.random.randn(2,2))
# 多次生成一模一样的数组
np.random.seed(8)
print('按指定随机种子，第一次生成随机数')
print(np.random.randint(1,5,(2,2)))
# 设置相同的随机种子
np.random.seed(8)
print('相同的种子，第二次生成：')
print(np.random.randint(1,5,(2,2)))


