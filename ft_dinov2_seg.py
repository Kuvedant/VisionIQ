import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from transformers import Dinov2Model, Dinov2ImageProcessor

# Step 1: Load the Pre-trained DINOv2 Model
processor = Dinov2ImageProcessor.from_pretrained("facebook/dino-v2-small")
model = Dinov2Model.from_pretrained("facebook/dino-v2-small")

# Step 2: Define Dataset Class
class UrbanSceneDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Step 3: Prepare Dataset and Dataloader
from PIL import Image
import glob

# Replace with your dataset paths
image_paths = glob.glob("/path/to/images/*.jpg")
mask_paths = glob.glob("/path/to/masks/*.png")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = UrbanSceneDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Step 4: Fine-Tune DINOv2
class DINOv2FineTuner(torch.nn.Module):
    def __init__(self, model):
        super(DINOv2FineTuner, self).__init__()
        self.backbone = model
        self.segmentation_head = torch.nn.Conv2d(384, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone(pixel_values=x).last_hidden_state
        logits = self.segmentation_head(features.permute(0, 2, 3, 1).contiguous())
        return logits

# Set number of classes for your dataset
num_classes = 21
fine_tuner = DINOv2FineTuner(model)
fine_tuner.train()

# Optimizer and Loss Function
optimizer = torch.optim.Adam(fine_tuner.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

# Training Loop
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tuner.to(device)

for epoch in range(epochs):
    epoch_loss = 0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = fine_tuner(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Step 5: Evaluate Model Using IoU
from sklearn.metrics import jaccard_score

def calculate_iou(preds, labels):
    preds = preds.argmax(dim=1).cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    return jaccard_score(labels, preds, average="weighted")

# Evaluate IoU on Validation Dataset
fine_tuner.eval()
iou_scores = []

with torch.no_grad():
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device, dtype=torch.long)
        outputs = fine_tuner(images)
        iou = calculate_iou(outputs, masks)
        iou_scores.append(iou)

print(f"Mean IoU: {np.mean(iou_scores):.4f}")

# Step 6: Visualize Results
def visualize_segmentation(image, mask, prediction):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(mask.cpu().numpy(), cmap="jet")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(prediction.argmax(dim=0).cpu().numpy(), cmap="jet")

    plt.show()

# Visualize Sample
sample_image, sample_mask = next(iter(dataloader))
sample_image, sample_mask = sample_image[0].to(device), sample_mask[0].to(device)
with torch.no_grad():
    prediction = fine_tuner(sample_image.unsqueeze(0))

visualize_segmentation(sample_image, sample_mask, prediction[0])
