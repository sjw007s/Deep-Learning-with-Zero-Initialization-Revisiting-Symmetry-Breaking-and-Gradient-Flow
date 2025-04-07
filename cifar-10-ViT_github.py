import torch
import torchvision
import csv
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
print(torch.cuda.current_device())

local_adress = "/proj/home/ibs/ccs/whvankoh/"

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

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
    
        self.a = nn.LayerNorm(dim)
        self.b = nn.Linear(dim, hidden_dim)
        self.c = nn.GELU()
        self.d = nn.Dropout(dropout)
        self.e = nn.Linear(hidden_dim, dim)
        self.f = nn.Dropout(dropout)
        torch.nn.init.zeros_(self.b.weight)
        torch.nn.init.zeros_(self.b.bias)
        
    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        x = self.e(x)
        x = self.f(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        torch.nn.init.zeros_(self.mlp_head.weight)
        torch.nn.init.zeros_(self.mlp_head.bias)
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

summary = list()
for i in range(10):
    model = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 500,
    depth = 6,
    heads = 10,
    mlp_dim = 2000,
    dropout = 0.0,
    emb_dropout = 0.0
    ).to("cuda")
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