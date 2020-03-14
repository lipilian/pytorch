# %%
import torch
torch.__version__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
x = torch.tensor(2., requires_grad=True)
y = x**2
y.backward()
# %% complex example 
u = torch.tensor(1.,requires_grad = True)
v = torch.tensor(2.,requires_grad = True)
f = u *v + u**2
f.backward()
print(u.grad)
print(v.grad)
# %% linspace get grad   
x = torch.linspace(-10, 10, 1000, requires_grad = True)
Y = torch.relu(x)
y = Y.sum()
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()
print(y)


# %%
