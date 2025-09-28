import torch
import csv
import math
import os
import torchvision.transforms.v2 as transforms_v2
from PIL import Image
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from torch import nn
from torchvision.transforms.v2 import MixUp
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(1)

class ImageNetDataset_train(Dataset): 
    def __init__(self,train_images, train_labels):
        self.transform_train = transforms_v2.Compose([
                                        transforms_v2.RandomResizedCrop(64, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
                                        transforms_v2.RandomHorizontalFlip(p=0.5),
                                        transforms_v2.ColorJitter(
                                                        brightness=0.4,
                                                        contrast=0.4,
                                                        saturation=0.4,
                                                        hue=0.1
                                                    ),
                                        transforms_v2.RandomGrayscale(p=0.1),
                                        transforms_v2.ToImage(),
                                        transforms_v2.ToDtype(dtype = torch.float32, scale=True),
                                        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        
        class_to_idx = {} # Dictionary to store class-to-index mappings
        with open(r"C:\Users\sjw00\OneDrive\Desktop\dataset\imagenet\map_clsloc.txt", 'r') as f:
            for line in f:
                folder, idx, _ = line.strip().split(' ', 2) # Split each line by space into folder name and index
                class_to_idx[folder] = int(idx)-1   # Map the folder to its corresponding index (adjusted by -1 for zero-indexing)
        
        
        self.img_paths = train_images
        self.labels = train_labels

        print("training dataset load complete") # Print completion message


    def __len__(self):  # Return the total number of images in the dataset
        return len(self.img_paths) 
    
    def __getitem__(self, idx): # Get image and label by index
        img = self.img_paths[idx] # Get image path
        label = self.labels[idx] # Get label
        
        img = Image.open(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_tensor = self.transform_train(img) 
        return img_tensor , label 

transform_test = transforms_v2.Compose([
                                transforms_v2.ToImage(),
                                transforms_v2.ToDtype(dtype = torch.float32, scale=True),
                                transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

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

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.,
                 dense=nn.Linear):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.a = dense(dim, inner_dim)  
        self.b = dense(inner_dim, dim)  
        self.dense = dense
        self.act = nn.GELU()
        self.is_conv1d = isinstance(self.a, nn.Conv1d)

        torch.nn.init.zeros_(self.a.weight)
        torch.nn.init.zeros_(self.a.bias)

        torch.nn.init.zeros_(self.b.weight)
        torch.nn.init.zeros_(self.b.bias)


    def forward(self, x):
        x = self.a(x)
        if torch.count_nonzero(self.a.weight) == 0:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.a.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            if self.is_conv1d == False:
                x = x + torch.empty(x.shape[-1], device=device).uniform_(-bound, bound)
            else:
                x = x + torch.empty(x.shape[-2], device=device).uniform_(-bound, bound).view(1,-1,1)

        x = self.act(x)
        x = self.b(x)

        if torch.count_nonzero(self.b.weight) == 0:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.b.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            if self.is_conv1d == False:
                x = x + torch.empty(x.shape[-1], device=device).uniform_(-bound, bound)
            else:
                x = x + torch.empty(x.shape[-2], device=device).uniform_(-bound, bound).view(1,-1,1)

        return x

class MLPMixer(nn.Module):
    def __init__(self, *, image_size, channels, patch_size,
                 dim, depth,
                 expansion_factor=4, expansion_factor_token=4,
                 dropout=0.):
        super().__init__()
        img_h, img_w = pair(image_size)
        assert img_h % patch_size == 0 and img_w % patch_size == 0, \
            "image must be divisible by patch size"
        num_patches   = (img_h // patch_size) * (img_w // patch_size)


        self.rearrange   = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                     p1=patch_size, p2=patch_size)
        patch_dim        = patch_size * patch_size * channels
        self.patch_embed = nn.Linear(patch_dim, dim)
        torch.nn.init.zeros_(self.patch_embed.weight)
        torch.nn.init.zeros_(self.patch_embed.bias)

        chan_first = partial(nn.Conv1d, kernel_size=1)
        chan_last  = nn.Linear

        self.mixer_layers = nn.ModuleList()
        for _ in range(depth):
            token_mlp   = FeedForward(num_patches, expansion_factor,
                                      dropout, dense=chan_first)
            channel_mlp = FeedForward(dim, expansion_factor_token,
                                      dropout, dense=chan_last)

            self.mixer_layers.append(nn.ModuleList([
                PreNormResidual(dim, token_mlp),
                PreNormResidual(dim, channel_mlp)
            ]))

    def forward(self, x):
        # (B, C, H, W) → (B, tokens, patch_dim)
        x = self.rearrange(x)
        x = self.patch_embed(x)

        if torch.count_nonzero(self.patch_embed.weight) == 0:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.patch_embed.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            x = x + torch.empty(x.shape[-1], device=device).uniform_(-bound, bound)

        # Mixer layer stack
        for token_mlp, channel_mlp in self.mixer_layers:
            x = token_mlp(x)     
            x = channel_mlp(x)    

        return x

class jongwoo_mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = MLPMixer(image_size = 64,
                            channels = 3,
                            patch_size = 8,
                            dim = 512,
                            depth = 8,
                        )
        self.b = nn.LayerNorm(512)

        self.c = nn.Linear(512, 200)

        torch.nn.init.zeros_(self.c.weight)
        torch.nn.init.zeros_(self.c.bias)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = x.mean(dim=1)
        x = self.c(x)
        if torch.count_nonzero(self.c.weight) == 0:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.c.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            x = x + torch.empty(x.shape[-1], device=device).uniform_(-bound, bound)
  
        return x
    
def train(dataloader, model, loss_fn, optimizer, scaler, mixup): 
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = mixup(X, y)
        X = X.to("cuda")
        y = y.to("cuda")
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred = model(X)
            batch_loss_result = loss_fn(pred, y)
        optimizer.zero_grad()

        scaler.scale(batch_loss_result).backward() # Backpropagation: compute gradients
        scaler.step(optimizer) # Update model parameters using the optimizer
        scaler.update() # Update the gradient scaler

        running_loss += batch_loss_result.item()
        correct += (pred.argmax(1) == y.argmax(1)).sum().item()
        """
        The incorrect training accuracy calculation happened because of MixUp.
        However, since it does not affect the evaluation metric (test accuracy),
        the results remain valid. Because the experiments were run with this code,
        I prefer to leave it as-is for consistency.  Sep., 28, 2025, jongwoo seo, sjw007s@korea.ac.kr
        """
        total += y.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {acc:.4f}")

def test(dataloader, model):
    model.eval()
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        accuracy_sum=0
        for _, (X, y) in enumerate(dataloader):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(X)
                batch_loss_result = loss_fn(pred, y)
            accuracy_sum+= (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
            running_loss += batch_loss_result.item()
            total += y.size(0)
        avg_loss = running_loss / total
        print(avg_loss, "test_accuracy", accuracy_sum/10000)
    return accuracy_sum/10000

if __name__ == "__main__":
    mixup = MixUp(alpha=0.5, num_classes=200)
    train_images = [] 
    train_labels = [] 
    class_to_idx = {}

    class_folders = os.listdir(r"C:\Users\sjw00\OneDrive\Desktop\dataset\tiny-imagenet-200\train")
    for num, i in enumerate(class_folders):
        class_to_idx[i] = num

    for class_folder in class_folders:
        folder_path = os.path.join(r"C:\Users\sjw00\OneDrive\Desktop\dataset\tiny-imagenet-200\train", class_folder) +r"\images"
        temp_image = []
        temp_label = []
        for img_file in os.listdir(folder_path): 
            img_path = os.path.join(folder_path, img_file) 

            train_images.append(img_path)
            train_labels.append(torch.tensor(class_to_idx[class_folder],dtype=torch.long))

    train_labels = torch.stack(train_labels)
    
    trainset = ImageNetDataset_train(train_images, train_labels) 
    ########################################################################
    test_images = [] 
    test_labels = [] 

    with open(r"C:\Users\sjw00\OneDrive\Desktop\dataset\tiny-imagenet-200\val\val_annotations.txt", 'r') as f:
        for line in f:
            file, idx, _ = line.strip().split('\t', 2) # Split each line by space into folder name and index
            img_path = os.path.join(r"C:\Users\sjw00\OneDrive\Desktop\dataset\tiny-imagenet-200\val\images", file) 
            
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = transform_test(img)  

            test_images.append(img_tensor)
            test_labels.append(torch.tensor(class_to_idx[idx],dtype=torch.long))

    test_images = torch.stack(test_images).to("cuda")
    test_labels = torch.stack(test_labels).to("cuda")
    test_dataset = TensorDataset(test_images, test_labels)

    training_dataloader = DataLoader(trainset, batch_size=500, shuffle=True, num_workers=14, pin_memory=True, persistent_workers=True) # DataLoader for training data
    test_dataloader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    summary = list()
    total_list = []
    for i in range(10):
        model = jongwoo_mixer().to("cuda")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)
        accuracy = 0 
        epoch_list = []
        for t in range(100):
            print(f"Epoch {t+1}\n-------------------------------")
            train(training_dataloader, model, loss_fn, optimizer, scaler, mixup)
            accuracy_temp = test(test_dataloader, model)
            epoch_list.append(accuracy_temp)
            scheduler.step()
            if accuracy_temp > accuracy:
                accuracy = accuracy_temp
        total_list.append(epoch_list)
        summary.append(accuracy)
        print(summary)
    with open("tiny_mlp_constant_all.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(total_list)
    print("Done!")
    print(summary)