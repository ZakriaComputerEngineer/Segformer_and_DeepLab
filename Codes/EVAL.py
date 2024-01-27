import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
# import matplotlib.pyplot as plt


class DeepLabV3Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Segmentation, self).__init__()

        # Use a pre-trained DeepLabV3 model with ResNet-50 backbone
        self.deeplabv3 = deeplabv3_resnet50(pretrained=True)
        self.deeplabv3.classifier = nn.Conv2d(2048, num_classes, kernel_size=1)

    def forward(self, x):
        return self.deeplabv3(x)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

#       THIS FUNCTION IS USED FOR RECONSTRUCTING THE GT IMAGES


def dilate_annotations(annotations_path, dilation_kernel_size):
    annotations = cv2.imread(annotations_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_annotations = cv2.dilate(annotations, kernel, iterations=1)
    return dilated_annotations


# Dataset class for your custom dataset
class CustomDataset(Dataset):
    def __init__(self, root, transform, transform_mask, ef=64, subset_fraction=0.2):
        self.dataset = ImageFolder(root, transform=transform)
        self.transform_mask = transform_mask
        self.enhancement_factor = ef
        self.subset_fraction = subset_fraction
        import random
        self.indices = random.sample(range(len(self.dataset)), int(
            subset_fraction * len(self.dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, _ = self.dataset[original_idx]
        mask_path = self.dataset.imgs[original_idx][0].replace("images", "gt")
        annotations_path = mask_path
        original_annotations = cv2.imread(
            annotations_path, cv2.IMREAD_GRAYSCALE)
        annotations_normalized = (original_annotations - np.min(original_annotations)) / (
            np.max(original_annotations) - np.min(original_annotations)) * 255
        dilation_kernel_size = 5
        kernel = np.ones(
            (dilation_kernel_size, dilation_kernel_size), np.uint8)
        dilated_annotations = cv2.dilate(
            annotations_normalized, kernel, iterations=1)
        dilated_annotations_pil = Image.fromarray(dilated_annotations)
        mask = self.transform_mask(dilated_annotations_pil)
        return img, mask


# GIVE DIRECTORY TO YOUR DATASET ROOT FOLDER
dataset_path = r"GIVEN DATASET PATH HERE"
custom_dataset = CustomDataset(
    root=dataset_path, transform=transform, transform_mask=transform_mask)
dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=True)

# NUM OF CLASSES IS 1 BECAUSE THERE IS EITHER OBJECT OR NO OBJECT
model = DeepLabV3Segmentation(num_classes=1)
#   CHOOSE FROM THESE LOSS FUNCTION
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
#   CHOOSE FROM THESE OPTIMIZER
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(r"YOUR MODEL FILE HERE"))
model.eval()

all_targets = []
all_predictions = []

# Testing loop
with torch.no_grad():
    for images, targets in tqdm(dataloader, desc='Testing'):
        try:
            outputs = model(images)
            predictions = torch.sigmoid(outputs['out'])
            loss = criterion(outputs['out'], targets)
            all_targets.extend(targets.view(-1).cpu().numpy().tolist())
            all_predictions.extend(predictions.view(-1).cpu().numpy().tolist())

            # UNCOMMENT THIS BLOCK 2 Visualize the result
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(outputs['out'][0, 0].cpu().numpy(), cmap='viridis')
            # plt.title('output')
            # plt.subplot(1, 2, 2)
            # plt.imshow(outputs.get('aux', None)[0, 0].cpu().numpy(), cmap='viridis')
            # plt.title('aux')
            # plt.show()
            # all_targets.extend(targets.view(-1).cpu().numpy().tolist())
            # all_predictions.extend(predictions.view(-1).cpu().numpy().tolist())
            # Visualize the result
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(predictions[0, 0].cpu().numpy(), cmap='viridis')
            # plt.title('MODEL OUTPUT')
            # plt.subplot(1, 2, 2)
            # plt.imshow(targets[0, 0].cpu().numpy(), cmap='viridis')
            # plt.title('ground truth')
            # plt.show()

        except Exception as e:
            print(f"Error processing batch: {e}")


# Convert targets to binary format
all_targets = [1 if t > 0.1 else 0 for t in all_targets]
all_predictions = [1 if p > 0.001 else 0 for p in all_predictions]

min_length = min(len(all_predictions), len(all_targets))
all_predictions = all_predictions[:min_length]
all_targets = all_targets[:min_length]

cm = confusion_matrix(all_targets, all_predictions)

accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)

# plotting
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', xticklabels=[
            'Background', 'Object'], yticklabels=['True', 'Actual'])
plt.title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%')
plt.show()

#   UNCOMMEN THIS TO VIEW SUMMARY
# Display model summary
# print(model)
