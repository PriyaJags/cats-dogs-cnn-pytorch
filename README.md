# Cats vs Dogs Image Classification using PyTorch

## Project Overview

This project explores image classification using PyTorch and Convolutional Neural Networks (CNNs). The goal is to classify images as either cats or dogs.

I implemented and compared three approaches:

1. **Custom CNN:** Built and trained a convolutional neural network from scratch.
2. **Transfer Learning – Feature Extraction:** Used a pretrained ResNet18 model and trained a new classification layer.
3. **Transfer Learning – Fine-Tuning:** Unfroze the final ResNet18 convolutional block (`layer4`) and fine-tuned it along with the new classification layer.

## Technologies Used

- Python
- PyTorch
- Torchvision
- CNNs
- ResNet18
- Transfer Learning
- TensorBoard

## Project Structure

```text
CNN_Project/
│
├── train_cnn.py
├── predict_image.py
├── transfer_learning.py
├── predict_transfer.py
├── fine_tuning.py
├── tensorboard_demo.py
│
├── cats_dogs/
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   └── test/
│       ├── cat/
│       └── dog/
│
└── .gitignore
```

The dataset and trained model files are not included in this repository.

## Model 1: Custom CNN

- Built a CNN using convolutional, ReLU and max-pooling layers.
- Trained the model using the cats-and-dogs dataset.
- Applied data augmentation using random horizontal flipping.
- Achieved approximately **71.75% test accuracy** in the improved run.

## Model 2: ResNet18 – Feature Extraction

- Used pretrained ResNet18 weights.
- Froze the pretrained convolutional layers.
- Replaced the final fully connected layer with a two-class classifier.
- Trained only the new classification layer.
- Achieved **99.50% test accuracy** in the recorded run.

## Model 3: ResNet18 – Fine-Tuning

- Started with pretrained ResNet18 weights.
- Unfroze the final convolutional block (`layer4`).
- Trained `layer4` and the new fully connected classification layer.
- Used different learning rates for the pretrained and newly added layers.
- Monitored training loss and accuracy using TensorBoard.

The recorded fine-tuning run achieved **98.75% test accuracy**.

These results are from individual runs and may vary between training sessions.

## TensorBoard

TensorBoard was used to monitor the training process.

The logged metrics include:
- Training Loss
- Training Accuracy

To launch TensorBoard:

```bash
tensorboard --logdir=runs
```

Then open `http://localhost:6006` in your browser.

## Key Learnings

Through this project, I practised:

- Building CNN architectures using PyTorch.
- Understanding convolution, pooling and feature maps.
- Working with datasets and DataLoaders.
- Training and evaluating deep learning models.
- Applying transfer learning and fine-tuning.
- Saving and loading model weights using `state_dict`.
- Using TensorBoard to visualise training metrics.

## Future Improvements

- Experiment with different CNN architectures.
- Explore additional data augmentation techniques.
- Compare model performance across repeated runs.
- Explore NLP and text classification using machine learning.
