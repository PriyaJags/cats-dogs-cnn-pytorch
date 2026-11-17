
from torch.utils.tensorboard import SummaryWriter

# Create TensorBoard log writer
writer = SummaryWriter("runs/cat_dog_experiment")

# Sample training results from 5 epochs
training_losses = [0.69, 0.45, 0.30, 0.18, 0.10]
training_accuracies = [55, 68, 78, 88, 94]

# Record loss and accuracy for each epoch
for epoch in range(5):
    writer.add_scalar("Training Loss", training_losses[epoch], epoch)
    writer.add_scalar("Training Accuracy", training_accuracies[epoch], epoch)

# Close the writer
writer.close()
