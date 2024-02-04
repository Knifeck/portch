import torch 
a= torch.randn(2,3)
print(a,a.type())

print(torch.rand(2,3))

print(torch.rand(2,3).type())

print(torch.eye(2,3))
print(torch.eye(2,3).type())