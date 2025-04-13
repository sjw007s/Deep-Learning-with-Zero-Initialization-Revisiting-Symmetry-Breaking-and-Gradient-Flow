import torch
import torchvision
import csv
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
#print(torch.cuda.current_device())

local_adress = "temp"

temp_list = list()
r=open(local_adress+'/cifar-10-python/training_data_CIFAR.csv', 'r' )
reader=csv.reader(r)
for target in reader:
    temp_list.append(target)
training_data = np.array(temp_list).reshape((-1, 3, 32, 32)).astype(np.float32)
print(training_data.shape)
tt = torch.from_numpy(training_data).to("cuda")
training_data = torch.zeros((50000, 3, 40, 40)).to("cuda")
training_data[:,:,4:36,4:36] = tt
temp_list = list()
r=open(local_adress+'/cifar-10-python/test_data_CIFAR.csv', 'r' )
reader=csv.reader(r)
for target in reader:
    temp_list.append(target)
test_data = np.array(temp_list).reshape((-1, 3, 32, 32)).astype(np.float32)
print(test_data.shape)
test_data = torch.from_numpy(test_data).to("cuda")

temp_list = list()
r=open(local_adress+'/cifar-10-python/training_target_CIFAR.csv', 'r' )
reader=csv.reader(r)
for target in reader:
    if target[0] == '0':
        temp_list.append([1,0,0,0,0,0,0,0,0,0])
        continue
    if target[0] == '1':
        temp_list.append([0,1,0,0,0,0,0,0,0,0])
        continue
    if target[0] == '2':
        temp_list.append([0,0,1,0,0,0,0,0,0,0])
        continue
    if target[0] == '3':
        temp_list.append([0,0,0,1,0,0,0,0,0,0])
        continue
    if target[0] == '4':
        temp_list.append([0,0,0,0,1,0,0,0,0,0])
        continue
    if target[0] == '5':
        temp_list.append([0,0,0,0,0,1,0,0,0,0])
        continue
    if target[0] == '6':
        temp_list.append([0,0,0,0,0,0,1,0,0,0])
        continue
    if target[0] == '7':
        temp_list.append([0,0,0,0,0,0,0,1,0,0])
        continue
    if target[0] == '8':
        temp_list.append([0,0,0,0,0,0,0,0,1,0])
        continue
    if target[0] == '9':
        temp_list.append([0,0,0,0,0,0,0,0,0,1])
        continue 
training_target = np.array(temp_list).astype(np.float32)
print(training_target.shape)
training_target = torch.from_numpy(training_target).to("cuda")

temp_list = list()
r=open(local_adress+'/cifar-10-python/test_target_CIFAR.csv', 'r' )
reader=csv.reader(r)
for target in reader:
    if target[0] == '0':
        temp_list.append([1,0,0,0,0,0,0,0,0,0])
        continue
    if target[0] == '1':
        temp_list.append([0,1,0,0,0,0,0,0,0,0])
        continue
    if target[0] == '2':
        temp_list.append([0,0,1,0,0,0,0,0,0,0])
        continue
    if target[0] == '3':
        temp_list.append([0,0,0,1,0,0,0,0,0,0])
        continue
    if target[0] == '4':
        temp_list.append([0,0,0,0,1,0,0,0,0,0])
        continue
    if target[0] == '5':
        temp_list.append([0,0,0,0,0,1,0,0,0,0])
        continue
    if target[0] == '6':
        temp_list.append([0,0,0,0,0,0,1,0,0,0])
        continue
    if target[0] == '7':
        temp_list.append([0,0,0,0,0,0,0,1,0,0])
        continue
    if target[0] == '8':
        temp_list.append([0,0,0,0,0,0,0,0,1,0])
        continue
    if target[0] == '9':
        temp_list.append([0,0,0,0,0,0,0,0,0,1])
        continue 
test_target = np.array(temp_list).astype(np.float32)
print(test_target.shape)
test_target = torch.from_numpy(test_target).to("cuda")

batch_size = 100
epochs = 500

training_dataset = TensorDataset(training_data, training_target)
test_dataset = TensorDataset(test_data, test_target)

training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        width_= torch.randint(0,8, (1,))
        height_= torch.randint(0,8, (1,))
        flip_=torch.randint(0,2,(1,))
        if flip_==0:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(X[:,:,height_:height_+32,width_:width_+32])
                batch_loss_result = loss_fn(pred, y)
        else:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(torch.flip(X[:,:,height_:height_+32,width_:width_+32],(3,)))
                batch_loss_result = loss_fn(pred, y)
        optimizer.zero_grad()
        batch_loss_result.backward()
        optimizer.step()
    
def test(dataloader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        accuracy_sum=0
        
        for batch, (X, y) in enumerate(dataloader):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(X)
                batch_loss_result = loss_fn(pred, y)
            accuracy_sum+= (torch.argmax(pred, dim=1) == torch.argmax(y,dim=1)).type(torch.float).sum().item()

    return accuracy_sum/10000

from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    a = dense(dim, inner_dim)
    b = dense(inner_dim, dim)

    torch.nn.init.zeros_(a.weight)
    torch.nn.init.zeros_(a.bias)

    return nn.Sequential(
        a,
        nn.GELU(),
        nn.Dropout(dropout),
        b,
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 4, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)]
    )

class jongwoo_mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = MLPMixer(image_size = 32,
                            channels = 3,
                            patch_size = 4,
                            dim = 512,
                            depth = 8,
                            num_classes = 100)
        self.b = nn.LayerNorm(512)
        self.c = nn.Linear(512, 10)
        torch.nn.init.zeros_(self.c.weight)
        torch.nn.init.zeros_(self.c.bias)
    def forward(self, x):
        #print("----------------value")
        x = self.a(x)
        x = self.b(x)
        x = x.mean(dim=1)
        x = self.c(x)
        return x

summary = list()
for i in range(10):
    model = jongwoo_mixer().to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    accuracy = 0 
    for t in range(epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        train(training_dataloader, model, loss_fn, optimizer)
        accuracy_temp = test(test_dataloader, model, loss_fn)
        scheduler.step()
        if accuracy_temp > accuracy:
            accuracy = accuracy_temp
    summary.append(accuracy)
    print(summary)
print("Done!")
