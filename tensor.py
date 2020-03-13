# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
torch.__version__

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


# %%
