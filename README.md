# Cat and Dog Image Classification using CNN and PyTorch

## Project Overview
This project uses a Convolutional Neural Network (CNN) built with PyTorch to classify images into two categories: cats and dogs.

The model learns visual patterns from labelled training images and predicts the class of new, unseen images.

## Technologies Used
- Python
- PyTorch
- Torchvision
- Pillow

## Dataset
- 1,000 training images
- 400 testing images
- Two classes: Cat and Dog
- Images resized to 64 × 64 pixels

## CNN Architecture
The model consists of:

1. Convolutional layer with 16 filters
2. ReLU activation
3. Max pooling
4. Convolutional layer with 32 filters
5. ReLU activation
6. Max pooling
7. Flatten layer
8. Fully connected layer with 64 neurons
9. Output layer with 2 classes

## Model Training
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Cross Entropy Loss
- Epochs: 15
- Batch size: 16

## Data Augmentation
Random horizontal flipping was applied to training images to introduce image variation and help the model generalize to unseen images.

Testing images were not augmented.

## Results
- Test accuracy: **71.75%**
- Correct predictions: 287 out of 400 test images

The trained model was also tested on a separate cat photograph and correctly classified it as a cat, with a predicted confidence of approximately 82.27%.

## Project Files
- `train_cnn.py` – Loads the dataset, builds and trains the CNN, evaluates it and saves the trained weights.
- `predict_image.py` – Loads the saved model and predicts the class of a new image.
- `organise_images.py` – Organises the downloaded images into class folders.
- `cat_dog_cnn.pth` – Saved model weights.
- `requirements.txt` – Required Python libraries.

## How to Run

1. Install the required libraries:

   `pip install -r requirements.txt`

2. Download and organise the dataset into the expected folder structure.

3. Train the model:

   `python train_cnn.py`

4. Place a test photograph in the project folder and update the image path in `predict_image.py`.

5. Run the prediction:

   `python predict_image.py`

## Key Learning Outcomes
- Understanding image tensors and RGB channels
- Building a CNN using `nn.Conv2d` and `nn.MaxPool2d`
- Training a neural network using backpropagation and an optimizer
- Evaluating model performance on unseen test data
- Applying data augmentation
- Saving and loading model weights using `state_dict()`
- Making predictions on new images

## Future Improvements
- Experiment with additional data augmentation techniques
- Increase the size and diversity of the dataset
- Explore deeper CNN architectures
- Compare performance with transfer learning models
