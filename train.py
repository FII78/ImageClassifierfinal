import argparse
import torch
from torch import nn, optim
from model import FlowerClassifier, load_checkpoint
from utils import load_data

def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # Load and preprocess data
    dataloader, class_to_idx = load_data(data_dir)

    # Build the model
    model = FlowerClassifier(
        input_size=25088,  # Update with the actual input size
        hidden_units=hidden_units,
        output_size=len(class_to_idx)
    )

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")

    # Save the model checkpoint
    model.class_to_idx = class_to_idx
    checkpoint = {
        'input_size': 25088,  # Update with the actual input size
        'hidden_units': hidden_units,
        'output_size': len(class_to_idx),
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a flower classifier network.")
    parser.add_argument("data_dir", help="Path to the data directory.")
    parser.add_argument("--save_dir", default="checkpoint.pth", help="Directory to save the checkpoint.")
    parser.add_argument("--arch", default="vgg13", help="Architecture of the pre-trained model.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training.")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training.")

    args = parser.parse_args()

    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
