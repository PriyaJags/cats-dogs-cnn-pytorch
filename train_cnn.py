
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------------
# 1. Find project folder
# ----------------------------------

project_folder = Path(__file__).resolve().parent

# ----------------------------------
# 2. Prepare image transformations
# ----------------------------------

# For training images: resize, randomly flip, and convert to tensor
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# For testing images: resize and convert to tensor only
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# ----------------------------------
# 3. Load training and testing images
# ----------------------------------

train_data = datasets.ImageFolder(
    root=project_folder / "cats_dogs" / "train",
    transform=train_transform
)

test_data = datasets.ImageFolder(
    root=project_folder / "cats_dogs" / "test",
    transform=test_transform
)

print("Training images:", len(train_data))
print("Testing images:", len(test_data))
print("Classes:", train_data.classes)

# ----------------------------------
# 4. Create DataLoaders
# ----------------------------------

train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=16,
    shuffle=False
)

# ----------------------------------
# 5. Define CNN model
# ----------------------------------

class CatDogCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

# ----------------------------------
# 6. Create model
# ----------------------------------

model = CatDogCNN()

print(model)

# ----------------------------------
# 7. Define loss function and optimizer
# ----------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ----------------------------------
# 8. Train the CNN
# ----------------------------------

epochs = 15

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )

print("Training completed!")

# 9 Evaluate the model on test images

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
# 10 Save the trained model

torch.save(model.state_dict(), project_folder / "cat_dog_cnn.pth")

print("Model saved successfully!")
