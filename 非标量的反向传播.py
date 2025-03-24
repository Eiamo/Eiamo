import torch
x=torch.tensor([2,3],dtype=torch.float,requires_grad=True)
j=torch.zeros(2,2)
y=torch.zeros(1,2)

print("原始的x，j，y是")
print(x,j,y)
y[0,0]=x[0]**2+3*x[1]
y[0,1]=x[1]**2+2*x[0]
y.backward(torch.tensor([[1,0]]),retain_graph=True)
j[0]=x.grad
x.grad=torch.zeros_like(x.grad)
y.backward(torch.tensor([[0,1]]),retain_graph=True)
j[1]=x.grad
print(j)