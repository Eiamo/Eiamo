import torch
x=torch.tensor([2])
w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)
y=torch.mul(w,x)
z=torch.add(y,b)
print("x,w,b的属性为：{}，{}，{}".format(x.requires_grad,w.requires_grad,b.requires_grad))
print("y,z的属性：{},{}".format(y.requires_grad,z.requires_grad))
print("x,w,b,y,z是否为叶子节点：{},{},{},{},{}".format(x.is_leaf,w.is_leaf,b.is_leaf,y.is_leaf,z.is_leaf))
z.backward()
print("w，b的梯度：{},{},{}".format(w.grad,b.grad,x.grad))
#print("y，z的梯度：{},{}".format(y.grad,z.grad))