#%%
import numpy as np
import torch
torch.__version__
# %%
x = torch.empty(5,3)
print(x)

# %%
x = torch.rand(5,3)
print(x)

# %%

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# %%

x = torch.tensor([5.5, 3])
print(x)

# %%
x = x.new_ones(5, 3, dtype=torch.double)      # new_* 方法来创建对象
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 覆盖 dtype!
print(x)             

# %%
print(x.size())


# %%
y = torch.rand(5, 3)
print(x + y)

# %%
print(torch.add(x, y))

# %%
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# %%
y.add_(x)
print(y)

# %%
print(x[:, 1])

# %%
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  #  size -1 从其他维度推断
print(x.size(), y.size(), z.size())

# %%

x = torch.randn(1)
print(x)
print(x.item())

# %%

a = torch.ones(5)
print(a)

#%%

b = a.numpy()
print(b)
# %%
a.add_(1)
print(a)
print(b)

# %%

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
    x = x.to(device)                       # 或者直接使用``.to("cuda")``将张量移动到cuda中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  

# %%


# %%
