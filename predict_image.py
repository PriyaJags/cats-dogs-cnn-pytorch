import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from PIL import Image

# 1. Find the project folder
project_folder = Path(__file__).resolve().parent

# 2. Prepare the same image transformation used during training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 3. Define the CNN model
# Recreate the CNN architecture because only the trained weights and biases
# were saved in cat_dog_cnn.pth. Load those learned weights into this model.
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

# 4. Load the saved model
model = CatDogCNN()

model.load_state_dict(
    torch.load(
        project_folder / "cat_dog_cnn.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# 5. Open your new photograph
image_path = project_folder / "my_cat.jpg.jpg"

image = Image.open(image_path).convert("RGB")

# 6. Transform the image into a tensor
image_tensor = transform(image)

# 7. Add a batch dimension
image_tensor = image_tensor.unsqueeze(0)

# 8. Make a prediction
with torch.no_grad():
    outputs = model(image_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()

# 9. Display the result
classes = ["cat", "dog"]

print("Predicted animal:", classes[predicted_class])
print("Confidence:", probabilities[0][predicted_class].item() * 100, "%")