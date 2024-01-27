#       THIS CODE IS FOR TRAINING THE DEEPLABV3 MODEL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Subset


class DeepLabV3Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Segmentation, self).__init__()
        self.deeplabv3 = deeplabv3_resnet50(pretrained=True)
        self.deeplabv3.classifier = nn.Conv2d(2048, num_classes, kernel_size=1)

    def forward(self, x):
        return self.deeplabv3(x)['out']


#       NORMALIZATION IS ONLY FOR RGB NOT GREYSCALE!
transform_input = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class SemanticSegmentationDataset(Dataset):
    def __init__(self, root, transform_input, transform_mask):
        self.dataset = ImageFolder(root, transform=None)
        self.transform_mask = transform_mask
        self.transform_input = transform_input

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        mask_path = self.dataset.imgs[idx][0].replace("images", "gt")
        mask = Image.open(mask_path).convert("L")
        # print("before trans:",mask.size)

        #       UNCOMMENT THIS TO RECONSTRUCT THE GT IMAGES

        # annotations_path = mask_path
        # original_annotations = cv2.imread(annotations_path, cv2.IMREAD_GRAYSCALE)
        # annotations_normalized = (original_annotations - np.min(original_annotations)) / (np.max(original_annotations) - np.min(original_annotations)) * 255
        # dilation_kernel_size = 5
        # kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        # dilated_annotations = cv2.dilate(annotations_normalized, kernel, iterations=1)
        # dilated_annotations_pil = Image.fromarray(dilated_annotations)
        # mask = self.transform_mask(dilated_annotations_pil)

        mask = self.transform_mask(mask)
        img = self.transform_input(img)

        # mask = mask.squeeze(0) if mask.dim() == 3 else mask
        # print("mask:",mask.shape,"\nimage:",img.shape)
        # input("Press Enter to continue...")

        return img, mask


def visualize(images, targets, predictions):
    images_np = images.cpu().numpy()
    targets_np = targets.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    plt.figure(figsize=(15, 5))
    for i in range(images_np.shape[0]):
        plt.subplot(3, images_np.shape[0], i + 1)
        plt.imshow(images_np[i, 0], cmap='gray')
        plt.title('Input')
        plt.subplot(3, images_np.shape[0], i + 1 + images_np.shape[0])
        plt.imshow(predictions_np[i, 0], cmap='gray')
        plt.title('Predicted')
        plt.subplot(3, images_np.shape[0], i + 1 + 2 * images_np.shape[0])
        plt.imshow(targets_np[i, 0], cmap='gray')
        plt.title('Ground Truth')
    plt.show()


def visualize_samples(model, dataloader, device):
    model.eval()
    images, targets = next(iter(dataloader))
    images, targets = images.to(device), targets.to(device)
    outputs = model(images)
    predictions = torch.argmax(F.softmax(outputs['out'], dim=1), dim=1)
    visualize(images, targets, predictions)


num_epochs = 2
batch_size = 4

# LOADING DATA, CALLING DATALOADER, 80-20 SPLIT
dataset_path = r"D:\abubakar\dataset"
semantic_dataset = SemanticSegmentationDataset(
    root=dataset_path, transform_input=transform_input, transform_mask=transform_mask)
total_size = len(semantic_dataset)
# total_size = int(0.1 * total_size)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(
    semantic_dataset, [train_size, val_size])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = DeepLabV3Segmentation(num_classes=1).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Count layers and parameters
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

        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}')

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for val_images, val_targets in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} (Validation)'):
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_targets)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(val_dataloader)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_val_loss}')

        # visualize_samples(model, val_dataloader, device)

    torch.save(model.state_dict(),
               r'D:\abubakar\models\deeplabv3_semantic_segmentation_model1.pth')
