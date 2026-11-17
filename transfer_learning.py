import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# ============================================================
# TRANSFER LEARNING: CAT vs DOG CLASSIFICATION USING RESNET18
# ============================================================
#
# PROJECT OVERVIEW:
# In this project, we used Transfer Learning with a pretrained
# ResNet18 model to classify images into two categories:
# Cat and Dog.
#
# WHAT IS TRANSFER LEARNING?
# Transfer Learning means using a model that has already been
# trained on a large dataset and adapting it to a new task.
#
# We used ResNet18, which was pretrained on ImageNet
# (1,000 image categories). It has already learned useful
# visual features such as edges, shapes, textures and patterns.
#
# WHAT WE CHANGED:
#
# 1. LOADED PRETRAINED RESNET18
#    - Loaded the ResNet18 architecture.
#    - Loaded its pretrained ImageNet weights.
#
# 2. FROZE THE PRETRAINED LAYERS
#    - Set requires_grad = False for all existing parameters.
#    - This prevented the pretrained layers from changing
#      during our initial training.
#
# 3. REPLACED THE FINAL LAYER
#    - Original ResNet18 output: 1,000 classes.
#    - Our task has only 2 classes: Cat and Dog.
#    - Replaced the original fully connected layer with:
#          nn.Linear(512, 2)
#    - 512 represents the input features from ResNet18.
#    - 2 represents the two output classes.
#
# 4. IMAGE PREPROCESSING
#    - Used the preprocessing transformations recommended
#      for the pretrained ResNet18 weights.
#    - Images were resized/cropped to 224 x 224 pixels,
#      converted to tensors and normalized using ImageNet
#      mean and standard deviation values.
#
# 5. LOSS FUNCTION AND OPTIMIZER
#    - Used CrossEntropyLoss for two-class classification.
#    - Used Adam optimizer with learning rate 0.001.
#    - Optimized only model.fc.parameters().
#    - Therefore, only the new final layer's weights and
#      biases were updated during training.
#
# 6. TRAINING
#    - Passed training images through ResNet18.
#    - Calculated prediction error using CrossEntropyLoss.
#    - Used backpropagation to calculate gradients.
#    - Adam updated the new final layer's parameters.
#    - The pretrained feature-extraction layers remained frozen.
#
# 7. MODEL EVALUATION
#    - Evaluated the trained model on separate test images.
#    - Used model.eval() and torch.no_grad() during testing.
#    - Calculated test accuracy.
#
# 8. SAVING AND PREDICTION
#    - Saved the trained model parameters using state_dict().
#    - For prediction, recreated the same ResNet18 architecture
#      and replaced its final layer with nn.Linear(512, 2).
#    - Loaded our saved Cat/Dog model weights.
#    - Used the model to classify new images.
#
# RESULTS:
#    - Test accuracy: 99.50% (398 correct out of 400 images).
#    - Successfully tested prediction on a new dog image.
#    - Dog prediction confidence: 98.63%.
#
# COMPARISON WITH OUR PREVIOUS CNN:
#    - Previous project: Built and trained our own CNN
#      architecture from scratch.
#    - Transfer Learning project: Reused ResNet18's pretrained
#      architecture and ImageNet weights.
#    - We trained only the new final layer instead of training
#      the entire network.
#
# IMPORTANT:
# This project uses FEATURE EXTRACTION, which is one approach
# to Transfer Learning.
#
# FINE-TUNING is a further step where we unfreeze selected
# pretrained layers and train them on our Cat/Dog dataset.
#
# ============================================================

# ------------------------------------------------
# 1. Load pretrained ResNet18
# ------------------------------------------------

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze all pretrained parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for 2 classes: Cat and Dog
model.fc = nn.Linear(512, 2)

# ------------------------------------------------
# 2. Prepare image transformations
# ------------------------------------------------

transform = weights.transforms()

# ------------------------------------------------
# 3. Load the datasets
# ------------------------------------------------

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

# ------------------------------------------------
# 4. Create DataLoaders
# ------------------------------------------------

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

# ------------------------------------------------
# 5. Define loss function and optimizer
# ------------------------------------------------

criterion = nn.CrossEntropyLoss()

# Train only the new final layer
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

# ------------------------------------------------
# 6. Train the model
# ------------------------------------------------

epochs = 5

for epoch in range(epochs):

    # Keep pretrained layers in evaluation mode
    model.eval()

    # Keep the new final layer in training mode
    model.fc.train()

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

        # Update final layer weights
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# ------------------------------------------------
# 7. Evaluate the model on test data
# ------------------------------------------------

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
accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
# Save the trained model
torch.save(
    model.state_dict(),
    "CNN_Project/resnet18_cat_dog.pth"
)

print("Model saved successfully!")
