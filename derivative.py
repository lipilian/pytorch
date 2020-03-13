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
# %%
