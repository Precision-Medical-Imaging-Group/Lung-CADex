import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import os

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")  # Convert grayscale to RGB
        label = self.img_labels.iloc[idx, -1]  # Assuming the label is in the last column
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Define transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 for CLIP
    transforms.ToTensor(),
])

# Assuming your CSV file is named 'data.csv' and is in the current directory
csv_file = 'data.csv'
img_dir = 'C:\Users\fu057938\test nodules'  
# Initialize dataset and dataloader
dataset = CustomImageDataset(annotations_file=csv_file, img_dir=img_dir, transform=transformations)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define a simple binary classification layer on top of CLIP
class CLIPClassifier(torch.nn.Module):
    def __init__(self, clip_model):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        # Assuming binary classification, adjust out_features for more classes
        self.classifier = torch.nn.Linear(in_features=clip_model.visual.projection_dim, out_features=1)
    
    def forward(self, images):
        image_features = self.clip_model.get_image_features(images)
        logits = self.classifier(image_features)
        return logits

# Initialize CLIP classifier
clip_classifier = CLIPClassifier(model)

# Define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
optimizer = torch.optim.Adam(clip_classifier.parameters(), lr=1e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    for images, labels in dataloader:
        # Convert labels to float and reshape to match output dimensions
        labels = labels.float().unsqueeze(1)
        
        # Forward pass
        outputs = clip_classifier(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Note: This code is for demonstration and might need adjustments based on your specific requirements.
