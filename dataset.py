#%%
from torch.utils.data import Dataset
from torchvision import transforms
#%% create the class of dataset 
class toy_set(Dataset):
    def __init__(self,length=100,transform=None):
        self.x = 2*torch.ones(length,2)
        self.y = torch.ones(length,1)
        self.len = length
        self.transform = transform
    def __getitem__(self,index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample 
    def __len__(self):
        return self.len
#%% create dataset 
dataset = toy_set()
print(len(dataset))
print(dataset[0])


# %% transform class 
class add_mult(object):
    def __init__(self,addx=1,muly=1):
        self.addx = addx
        self.muly = muly
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x,y
        return sample


# %% apply transform to dataset 
a_m = add_mult()
x_,y_ = a_m(dataset[0])
print(x_,y_)

# %% add transform to dataset 
dataset_ = toy_set(transform=a_m)
dataset_[1]

# %% transform compose 
class mult(object):
    def __init__(self,mul = 100):
        self.mul = mul
    def __call__(self,sample):
        x = sample[0]
        y = sample[1]
        x = x*self.mul
        y = y * self.mul
        sample = x,y
        return sample
# %% combine two transform 
data_transform = transforms.Compose([add_mult(),mult()])
x_,y_ = data_transform(dataset[0])
print(x_,y_)
dataset_ = toy_set(transform=data_transform)
dataset_[0]
# %%
