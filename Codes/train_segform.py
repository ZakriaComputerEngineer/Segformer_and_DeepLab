# THIS CODE IS FOR TRAINING SEGFORMED MODEL

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Subset
from segformer.model.segformer import Segformer

class SegFormerWithDeepLabBackbone(nn.Module):
    def __init__(self, num_classes):
        super(SegFormerWithDeepLabBackbone, self).__init__()

        # Use a pre-trained DeepLabV3 model with ResNet-50 backbone
        deeplabv3_model = deeplabv3_resnet50(pretrained=True)
        deeplabv3_backbone = nn.Sequential(*list(deeplabv3_model.children())[:-1])
        self.segformer = Segformer(
            image_size=(256, 256),
            patch_size=4,
            in_channels=2048,
            embed_dim=768,
            num_heads=12,
            num_encoder_layers=12,
            num_decoder_layers=12,
            num_classes=num_classes,
        )
        self.segformer.encoder = deeplabv3_backbone

    def forward(self, x):
        return self.segformer(x)

import torch.optim as optim

model = SegFormerWithDeepLabBackbone(num_classes=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Count layers and parameters
num_layers = len(list(model.parameters()))
num_parameters = sum(p.numel() for p in model.parameters())
print(f"Number of layers: {num_layers}")
print(f"Number of parameters: {num_parameters}")
print(model)
#from torchsummary import summary
# Print the model summary
#summary(model, (3, 224, 224))
#trainging loo

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # DataLoader for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = DeepLabV3Segmentation(num_classes=1).to(device)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_layers = len(list(model.parameters()))
    num_parameters = sum(p.numel() for p in model.parameters())

    print(f"Number of layers: {num_layers}")
    print(f"Number of parameters: {num_parameters}")

    #   TRAINING LOOP

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, targets in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} (Training)'):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            #print(total_loss)

        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}')
        
        #   VALIDATION LOOP
        
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for val_images, val_targets in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} (Validation)'):
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_targets)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(val_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_val_loss}')
        
        #visualize_samples(model, val_dataloader, device)

    torch.save(model.state_dict(), r'D:\abubakar\models\deeplabv3_semantic_segmentation_model1.pth')