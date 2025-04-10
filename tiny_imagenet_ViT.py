import torch
import csv
import os
import torchvision.transforms.v2 as transforms_v2
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from torch import nn
from torchvision.transforms.v2 import MixUp
from einops.layers.torch import Rearrange
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
        torch.nn.init.zeros_(self.e.weight)
        
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
        model = model = ViT(
                            image_size = 64,
                            patch_size = 8,
                            num_classes = 200,
                            dim = 500,
                            depth = 6,
                            heads = 10,
                            mlp_dim = 2000,
                            dropout = 0.0,
                            emb_dropout = 0.0
                            ).to("cuda")
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)
        accuracy = 0 
        epoch_list = []
        for t in range(50):
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
    with open("tiny_ViT_zero.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(total_list)
    print("Done!")
    print(summary)