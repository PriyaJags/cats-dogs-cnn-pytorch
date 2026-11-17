
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# FINE-TUNING RESNET18: CAT vs DOG CLASSIFICATION
# ============================================================
#
# PROJECT OVERVIEW:
# We use Transfer Learning with pretrained ResNet18 weights.
#
# FEATURE EXTRACTION:
# Previously, we froze all pretrained layers and trained
# only the new final classification layer (fc).
#
# FINE-TUNING:
# In this project, we unfreeze the last ResNet block (layer4)
# as well as the new final layer (fc).
#
# This allows the model to adapt some of its pretrained
# visual features to our Cat vs Dog classification task.
#
# TRAINABLE LAYERS:
#   layer1, layer2, layer3 = Frozen
#   layer4                = Trainable
#   fc                    = Trainable
#
# LOSS FUNCTION:
# CrossEntropyLoss measures the classification error.
#
# OPTIMIZER:
# Adam updates layer4 and fc using different learning rates.
#
# TENSORBOARD:
# Records training loss and training accuracy so we can
# monitor the model's learning through graphs.
#
# ============================================================


# 1. Create TensorBoard writer
writer = SummaryWriter("runs/resnet18_finetuning")


# ------------------------------------------------------------
# 2. Load pretrained ResNet18
# ------------------------------------------------------------

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze all pretrained parameters initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last ResNet block
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace the original 1000-class layer with 2 outputs
model.fc = nn.Linear(512, 2)

# Check which parameters are trainable
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, "is trainable")


# ------------------------------------------------------------
# 3. Prepare image transformations
# ------------------------------------------------------------

# Use preprocessing recommended for pretrained ResNet18
transform = weights.transforms()


# ------------------------------------------------------------
# 4. Load the datasets
# ------------------------------------------------------------

train_dataset = datasets.ImageFolder(
    "CNN_Project/cats_dogs/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "CNN_Project/cats_dogs/test",
    transform=transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))


# ------------------------------------------------------------
# 5. Create DataLoaders
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# ------------------------------------------------------------
# 6. Define loss function and optimizer
# ------------------------------------------------------------

criterion = nn.CrossEntropyLoss()

# Use different learning rates for different layers
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 0.0001},
    {"params": model.fc.parameters(), "lr": 0.001}
])


# ------------------------------------------------------------
# 7. Train the model
# ------------------------------------------------------------

epochs = 5

for epoch in range(epochs):

    # Set the model to training mode
    model.train()

    # Keep frozen earlier blocks' BatchNorm statistics unchanged
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    # Keep the trainable blocks in training mode
    model.layer4.train()
    model.fc.train()

    running_loss = 0.0

    # Reset accuracy counters for each epoch
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass: get model predictions
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation: calculate gradients
        loss.backward()

        # Update trainable weights in layer4 and fc
        optimizer.step()

        # Accumulate training loss
        running_loss += loss.item()

        # Find the predicted class for each image
        _, predicted = torch.max(outputs, 1)

        # Count correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss for this epoch
    epoch_loss = running_loss / len(train_loader)

    # Calculate training accuracy for this epoch
    epoch_accuracy = 100 * correct / total

    # Record training loss in TensorBoard
    writer.add_scalar("Training Loss", epoch_loss, epoch + 1)

    # Record training accuracy in TensorBoard
    writer.add_scalar("Training Accuracy", epoch_accuracy, epoch + 1)

    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Loss: {epoch_loss:.4f}, "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# ------------------------------------------------------------
# 8. Evaluate the model on test data
# ------------------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        # Make predictions
        outputs = model(images)

        # Find predicted class
        _, predicted = torch.max(outputs, 1)

        # Count correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate test accuracy
test_accuracy = 100 * correct / total

print(f"Test Accuracy: {test_accuracy:.2f}%")


# ------------------------------------------------------------
# 9. Save the fine-tuned model
# ------------------------------------------------------------

torch.save(
    model.state_dict(),
    "CNN_Project/resnet18_cat_dog_finetuned.pth"
)

print("Fine-tuned model saved successfully!")


# ------------------------------------------------------------
# 10. Close TensorBoard writer
# ------------------------------------------------------------

writer.close()

print("TensorBoard logs saved successfully!")
