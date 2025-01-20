"""
My computer system specification
i9-10900X
RAM 64GB
Samsung SSD 970 PRO 512GB
RTX 3090 2 units

Window 11
Pytorch 2.5.1
Anaconda3-2024.10-1-Windows-x86_64
cudnn-windows-x86_64-8.9.7.29_cuda12-archive
cuda_12.4.0_windows_network

Email: sjw007s@korea.ac.kr
"""

import csv
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = "cuda:1" 

def initialize_csv(file_name):
    if os.path.exists(file_name):
        return  
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Test Loss", "Top-1 Accuracy", "Top-5 Accuracy", "Max Accuracy-1", "Max Accuracy-5"])

def append_to_csv(file_name, epoch, train_loss, test_loss, top1_accuracy, top5_accuracy, max_accuracy_1, max_accuracy_5):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, test_loss, top1_accuracy, top5_accuracy, max_accuracy_1, max_accuracy_5])
    
def format_time(start_time):
    """Convert seconds to a formatted string: HH:MM:SS."""
    epoch_duration = time.time() - start_time
    hours = int(epoch_duration // 3600)
    minutes = int((epoch_duration % 3600) // 60)
    seconds = int(epoch_duration % 60)
    print(f"{hours:02}:{minutes:02}:{seconds:02}")

# Parsing a mapping file (reading a text file) from 2017 ILSVRC kit for target label
def train_parse_mapping_file(mapping_file):
    class_to_idx = {} # Dictionary to store class-to-index mappings
    with open(mapping_file, 'r') as f:
        for line in f:
            folder, idx, _ = line.strip().split(' ', 2) # Split each line by space into folder name and index
            class_to_idx[folder] = int(idx)-1   # Map the folder to its corresponding index (adjusted by -1 for zero-indexing)
    return class_to_idx

# Parsing validation ground truth file
def test_parse_mapping_file(mapping_file):
    class_to_idx = [] # List to store validation labels
    with open(mapping_file, 'r') as f:
        for line in f:
            number = line.strip() # Read each line and strip any extra whitespace
            class_to_idx.append(int(number)-1) # Append the class index to the list (adjusted by -1)
    return class_to_idx

# training data augmentation
transform_train = transforms_v2.Compose([
    transforms_v2.ToDtype(dtype = torch.float32, scale=True),
    transforms_v2.RandomResize(min_size=256, max_size=481), # Randomly resize image between [256, 481) pixels 
    transforms_v2.RandomHorizontalFlip(p=0.5), # 50% chance of horizontally flipping the image
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet mean and std
    transforms_v2.RandomCrop(224) # Randomly crop to 224x224
])

# test data augmentation
transform_test = transforms_v2.Compose([
    transforms_v2.ToDtype(dtype = torch.float32, scale=True),
    transforms_v2.Resize(256),  # Resize the shorter side to 256 pixels
    transforms_v2.CenterCrop(256), # Center crop the image to 256x256
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize
    transforms_v2.TenCrop(224) # Apply ten-crop augmentation (corner and center crops)
])

# training dataset
class ImageNetDataset_train(Dataset): 
    def __init__(self, root_dir, mapping_file, transform):
        self.transform = transform # Transformations to apply to each image
        self.totensor = transforms_v2.ToImage() # Transform to convert data into tensor format
        
        class_to_idx = train_parse_mapping_file(mapping_file) # Parse mapping file for class indices
        
        self.img_paths = [] # List to store image file paths
        self.labels = [] # List to store labels

        class_folders = os.listdir(root_dir)
        for class_folder in class_folders:
            folder_path = os.path.join(root_dir, class_folder) # Get full folder path
            for img_file in os.listdir(folder_path): # Iterate through images in the folder
                img_path = os.path.join(folder_path, img_file) # Get full image path
                
                label = class_to_idx[class_folder] # Get label from class_to_idx mapping

                self.img_paths.append(img_path) # Add image path to list
                self.labels.append(label) # Add label to list
 
        self.labels = torch.tensor(self.labels, dtype=torch.long) # Convert labels to tensor

        print("training dataset load complete") # Print completion message


    def __len__(self):  # Return the total number of images in the dataset
        return len(self.img_paths) 
    
    def __getitem__(self, idx): # Get image and label by index
        img_path = self.img_paths[idx] # Get image path
        label = self.labels[idx] # Get label
        
        img = Image.open(img_path) # Open image
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = self.totensor(img).to(device)  # Convert image to tensor and move to GPU
        img_tensor = self.transform(img_tensor) # Apply transformations to the image
        
        return img_tensor, label.to(device) # Return image and label (moved to GPU)
        
# test dataset
class ImageNetDataset_test(Dataset):
    def __init__(self, root_dir, mapping_file, transform):
        self.transform = transform
        self.totensor = transforms_v2.ToImage()

        self.img_paths = []
        for img_file in sorted(os.listdir(root_dir)): # Scan images in sorted order
            img_path = os.path.join(root_dir, img_file)
            self.img_paths.append(img_path)

        self.labels = test_parse_mapping_file(mapping_file)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        print("test dataset load complete")

    def __len__(self): 
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path) # Open image
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = self.totensor(img).to(device)  # Convert image to tensor and move to GPU
        img_tensor = self.transform(img_tensor) # Apply transformations to the image
        
        return img_tensor, label.to(device) 

def test_collate(batch): # Custom collate function for batching test data
    imgs, labels_original = zip(*batch)   # Unzip batch into images and labels
    imgs = list(imgs) # Convert to list for stacking
    for i in range(10):
        imgs[i] = torch.stack(imgs[i]) # Stack ten-crop images
    imgs = torch.stack(imgs) # Stack into final batch
    imgs = imgs.reshape(100, 3, 224, 224) # Reshape to batch size 500
    
    labels_original = torch.stack(labels_original) # Stack labels
    labels = torch.repeat_interleave(labels_original, 10) # Repeat labels for ten-crop

    return imgs, labels, labels_original

class CNN(nn.Module): # Define a neural network model by inheriting from nn.Module
    def __init__(self):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(3 * 224 * 224, 512)
        self.fc2 = nn.Linear(512, 1000)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(dataloader, model, loss_fn, optimizer):
    model.train() # Set the model to training mode
    loss_sum=0 # Initialize the cumulative loss
    
    # Iterate through the training data
    for X, y in dataloader:     
        optimizer.zero_grad() # Reset gradients before backpropagation
        pred = model(X) # Forward pass: generate predictions
        batch_loss_result = loss_fn(pred, y) # Compute the loss for the batch
        batch_loss_result.backward() # Backpropagation: compute gradients
        optimizer.step() # Update model parameters using the optimizer
        loss_sum+=batch_loss_result.item() # Accumulate batch loss

    loss_sum=loss_sum/len(dataloader.dataset) # Normalize the total loss by the dataset size
    return loss_sum

def test(dataloader, model, loss_fn):
    model.eval() # Set the model to evaluation mode
    
    with torch.no_grad(): # Disable gradient calculations for efficiency
        loss_sum = 0 # Initialize cumulative loss
        top1_correct = 0 # Initialize Top-1 accuracy counter
        top5_correct = 0 # Initialize Top-5 accuracy counter
        total_samples = 50_000 # Total number of test samples
        batch_size = 10 # Number of samples in each batch
 
        # Iterate through the test data
        for X, y, y_original in dataloader:
            pred = model(X)  # Forward pass: generate predictions
            
            batch_loss_result = loss_fn(pred, y) # Compute batch loss
            loss_sum += batch_loss_result.item() # Accumulate batch loss
      
            softmax_pred = torch.softmax(pred, dim=1) # Apply softmax to get probabilities
            view_softmax_pred = softmax_pred.view(batch_size, 10, 1000) # Reshape for Top-1/Top-5 calculations
            mean_view_softmax_pred = view_softmax_pred.mean(dim=1) # Take mean across a specific axis
            top1_pred = torch.argmax(mean_view_softmax_pred, dim=1) # Get Top-1 predictions

            top1_correct += (top1_pred == y_original).sum().item() # Count correct Top-1 predictions
         
            _, top5_pred = torch.topk(mean_view_softmax_pred, 5, dim=1)  # Get Top-5 predictions
       
            top1_y_expanded = y_original.view(-1, 1)  # Expand ground truth for Top-5 comparison
      
            top5_correct += torch.sum(torch.any(torch.eq(top5_pred, top1_y_expanded),dim=1)).item() # Count correct Top-5 predictions
            
        avg_loss = loss_sum / (10*total_samples) # Normalize loss over total test samples
        top1_accuracy = 100*top1_correct / total_samples # Compute Top-1 accuracy
        top5_accuracy = 100*top5_correct / total_samples # Compute Top-5 accuracy

    return avg_loss, top1_accuracy, top5_accuracy # Return average loss, Top-1 accuracy, and Top-5 accuracy

if __name__ == "__main__":
    # GPU setting
    print("GPU device currently in use:", device) # Print the current GPU device

    train_dir = r"C:\Users\sjw00\OneDrive\Desktop\dataset\imagenet\ILSVRC2012_img_train"  # training data location
    train_mapping_file = r"C:\Users\sjw00\OneDrive\Desktop\dataset\imagenet\map_clsloc.txt"  # training data mapping file location
    trainset = ImageNetDataset_train(root_dir=train_dir, mapping_file=train_mapping_file, transform=transform_train) 

    test_dir = r"C:\Users\sjw00\OneDrive\Desktop\dataset\imagenet\ILSVRC2012_img_val"  # test data location
    test_mapping_file = r"C:\Users\sjw00\OneDrive\Desktop\dataset\imagenet\ILSVRC2012_validation_ground_truth.txt"  # test data target label location
    testset = ImageNetDataset_test(root_dir=test_dir, mapping_file = test_mapping_file, transform=transform_test)  
    
    train_dataloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=5, persistent_workers=True) # DataLoader for training data
    test_dataloader = DataLoader(testset, batch_size=10, shuffle=False, collate_fn = test_collate, num_workers=2, persistent_workers=True) 

    loss_fn = nn.CrossEntropyLoss(reduction='sum') # Define the loss function

    # Main training and evaluation loop
    for i in range(1):
        model = CNN().to(device)  # Initialize and move the model to GPU
        initialize_csv(f"{model.__class__.__name__}_{device[-1]}_{i}.csv")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Define the optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.95) # Learning rate scheduler
        
        top1_accuracy = 0 # Initialize best Top-1 accuracy
        top5_accuracy = 0 # Initialize best Top-5 accuracy

        epochs = 5 # Number of epochs
        for epoch in range(epochs):
            start_time = time.time() # Record the start time of the epoch

            train_loss = train(train_dataloader, model, loss_fn, optimizer) # Train the model
            test_loss, top1_temp, top5_temp = test(test_dataloader, model, loss_fn) # Evaluate the model

            scheduler.step() # Update the learning rate scheduler

            format_time(start_time) # Calculate epoch duration

            # Update best Top-1 accuracy
            if top1_temp > top1_accuracy:
                top1_accuracy = top1_temp

            # Update best Top-5 accuracy
            if top5_temp > top5_accuracy:
                top5_accuracy = top5_temp

            print(train_loss, test_loss, top1_temp, top5_temp, top1_accuracy, top5_accuracy)

            # Store results for analysis
            append_to_csv(f"{model.__class__.__name__}_{device[-1]}_{i}.csv", epoch, train_loss, test_loss, top1_temp, top5_temp, top1_accuracy, top5_accuracy)


    print("Done!") # Indicate the end of training and evaluation

