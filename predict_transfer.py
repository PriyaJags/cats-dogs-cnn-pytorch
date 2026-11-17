
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

# 1. Load the pretrained ResNet18 architecture
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# 2. Replace the final layer with 2 outputs
model.fc = nn.Linear(512, 2)

# 3. Load our saved trained weights
model.load_state_dict(
    torch.load(
        "CNN_Project/resnet18_cat_dog.pth",
        map_location="cpu",
        weights_only=True
    )
)

# 4. Set model to evaluation mode
model.eval()

# 5. Prepare image transformation
transform = weights.transforms()

# 6. Load a new image
image = Image.open("CNN_Project/my_dog.jpg").convert("RGB")

# 7. Transform image and add batch dimension
image = transform(image).unsqueeze(0)

# 8. Make prediction
with torch.no_grad():
    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

# 9. Display prediction
classes = ["cat", "dog"]

print("Predicted class:", classes[predicted.item()])
print("Confidence:", f"{confidence.item() * 100:.2f}%")
