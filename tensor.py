# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
torch.__version__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %% [markdown]
# Tensor operation

# %%
a = torch.tensor([7,4,3,2,6]) # Create the Tensor
print(a[0]) # access the data
print(a.dtype) # find the type of data store in the tensor (int , double ,float)
b = torch.tensor([0.0,1.0]) # creat float tensor
print(b.dtype)
print(b.type())

# %% change the type of tensor
c = a.type(torch.float64)
print(c.dtype)

# %% get size and dimensions of 
print(a.size())
print(a.ndimension())

# %% convert 1D to 2D
c = a.view(5,1)
c = a.view(-1,1)

# %% numpy convert 
numpy_array = np.array([0.0, 1.0, 2.0])
torch_tensor = torch.from_numpy(numpy_array)
back_to_numpy = torch_tensor.numpy()

# %% pandas_series convert 
pandas_series = pd.Series([0.1,2,0.3])
pandas_to_torch = torch.from_numpy(pandas_series.values)

# %% convert tensor to list 
torch_to_list = a.tolist()
print(torch_to_list)

# %% return number
a[0].item()

# %% edit tensor 
a[0] = 9
print(a[0:3])
a[0:2] = torch.tensor([100,200])

# %% Vector Addition , multiplication
u = torch.tensor([1,2])
v = torch.tensor([2,3])
z = u + v
print(z)
z = 2 * u 
print(z)
print(u * v)
print(torch.dot(u,v))
print(u + 1)

# %% mean max  
u = u.type(torch.float64)
print(u.mean())
print(u.max())
# %% 
torch.sin(u)
torch.linspace(-1,2,steps=5)

# %%
x = torch.linspace(0, 2*np.pi, 100)
y = torch.sin(x)
%matplotlib inline    
plt.plot(x.numpy(),y.numpy())

# %% 2D tensor    
x = torch.tensor([[2,1],[1,2]])
y = torch.tensor([[2,1],[1,2]])
print(x * y)
print(x[0:2,1])

# %% 2D tensor matrix multiplication 
print(torch.mm(x,x))
# %%
