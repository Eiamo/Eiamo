import torch


x=torch.ones(2,requires_grad=True)
y=x**2+3
c=y.detach()
z=c*x
z.sum().backward()
x.grad==c
print(x)
print(x.grad)
c.grad_fn==None
c.requires_grad

x.grad.zero_()
print(x.grad)

y.sum().backward()
print(x.grad==2*x)