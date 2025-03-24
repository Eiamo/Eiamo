import torch
# 设置一个随机种子
torch.manual_seed(100)
x=torch.randn(2,3)
# print(x)
# print("x的第一行数据")
# print(x[0,:])
# print("x的第一列数据")
# print(x[:,0])
# 生成是否大于0的byter张量
mask=x>0
# 获取大于0的值
torch.masked_select(x,mask)
# 获取下标
torch.nonzero(mask)
# 获取指定索引对应的值
index=torch.LongTensor([[0,1,1]])
torch.gather(x,0,index)
index=torch.LongTensor([[0,1,1],[1,1,1]])
a=torch.gather(x,1,index)
# 把a的返回值回到一个2*3的o阵中
z=torch.zeros(2,3)
z.scatter_(1,index,a)
print('z最终是',z)
print('a最终是',a)


