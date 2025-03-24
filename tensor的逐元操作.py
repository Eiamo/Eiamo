import torch
t=torch.randn(1,3)
t1=torch.randn(3,1)
t2=torch.randn(1,3)
print(torch.addcdiv(t,t1,t2,value=0.1))
torch.sigmoid(t)
# 把t限制在0 1之间
torch.clamp(t,0,1)
print(t)