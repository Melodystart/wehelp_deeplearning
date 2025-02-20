import random
import torch
def random_weight(x, y, z):
  w1 = []
  w2 = []
  b1 = []
  b2 = []
  for i in range(y):
    a = []
    b1.append(random.uniform(-1, 1))
    for j in range(x):
      a.append(random.uniform(-1, 1))
    w1.append(a)

  for k in range(z):
    b = []
    b2.append(random.uniform(-1, 1))
    for i in range(y):
      b.append(random.uniform(-1, 1))
    w2.append(b)
  return w1, w2, b1, b2

# w1, w2, b1, b2 = random_weight(2, 2, 1)

print("------ Task 3-1 ------")
data = [[2, 3, 1], [5, -2, 1]]
tensor = torch.tensor(data)
print(tensor.dtype, tensor.shape)
print("------ Task 3-2 ------")
shape = (3,4,2)
rand_tensor = torch.rand(shape)
print(rand_tensor.shape)
print(rand_tensor)
print("------ Task 3-3 ------")
shape = (2,1,5)
ones_tensor = torch.ones(shape)
print(ones_tensor.shape)
print(ones_tensor)
print("------ Task 3-4 ------")
mat1 = torch.tensor([[1, 2, 4], [2, 1, 3]])
mat2 = torch.tensor([[5], [2], [1]])
matmul_tensor = torch.matmul(mat1, mat2)
print(matmul_tensor.shape)
print(matmul_tensor)
print("------ Task 3-5 ------")
mat1 = torch.tensor([[1, 2], [2, 3], [-1, 3]])
mat2 = torch.tensor([[5, 4], [2, 1], [1, -5]])
mul_tensor = mat1*mat2
print(mul_tensor.shape)
print(mul_tensor)
