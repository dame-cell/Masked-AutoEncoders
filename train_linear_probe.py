import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from datasets import load_dataset
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import math
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from configuration import MAEConfig
from utils import setup_seed, count_parameters
from torch.optim.lr_scheduler import LambdaLR
from modeling_mae import MAE_ViT, ViT_Classifier

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="damerajee/MAE", filename="model.pt", local_dir="model")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Linear Probe on CIFAR-10 with MAE_ViT encoder")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs (default: 2)")
    parser.add_argument('--pretrained', type=bool, default=True, help="Whether to use a pre-trained model or not")
    parser.add_argument('--path_to_model', type=str,default=None, help="If pretrained, pass path to model")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument('--batch_size', type=int, default=12, help="Batch size for training and validation (default: 12)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for optimizer (default: 1e-4)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Directory for saving model checkpoints
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    setup_seed(seed=args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)

    # Define CIFAR-10 data transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  
    ])

    # Load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load pre-trained MAE model (assuming it's already loaded)
    mae_model = MAE_ViT(config=MAEConfig())  # Use your pre-trained MAE model
    if args.pretrained and args.path_to_model is not None:        
        mae_model.load_state_dict(torch.load(args.path_to_model))
    linear_probe = ViT_Classifier(model=mae_model, num_classes=10).to(DEVICE)

    print("Model Parameters:", count_parameters(linear_probe))

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(linear_probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    lr_func = lambda epoch: min((epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / args.epochs * math.pi) + 1))
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    print("Starting to train")

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(args.epochs):
        linear_probe.train()
        epoch_train_losses = []
        epoch_train_accuracies = []

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, (image, labels) in progress_bar:
            start_time = time.time()
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass through linear probe
            logits = linear_probe(image)
            loss = criterion(logits, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Calculate accuracy using sklearn
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            accuracy = accuracy_score(labels_np, preds)

            epoch_train_losses.append(loss.item())
            epoch_train_accuracies.append(accuracy)

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'accuracy': f"{accuracy * 100:.2f}%",
                'lr': f"{lr_scheduler.get_last_lr()[0]:.6f}"
            })

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        avg_train_accuracy = sum(epoch_train_accuracies) / len(epoch_train_accuracies)

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # Validation step
        linear_probe.eval()
        epoch_val_losses = []
        epoch_val_accuracies = []

        with torch.no_grad():
            val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation {epoch+1}/{args.epochs}")
            
            for val_step, (val_image, val_labels) in val_progress_bar:
                val_image = val_image.to(DEVICE)
                val_labels = val_labels.to(DEVICE)

                logits = linear_probe(val_image)
                val_loss = criterion(logits, val_labels)

                # Calculate accuracy using sklearn
                val_preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_labels_np = val_labels.cpu().numpy()
                val_accuracy = accuracy_score(val_labels_np, val_preds)

                epoch_val_losses.append(val_loss.item())
                epoch_val_accuracies.append(val_accuracy)

            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            avg_val_accuracy = sum(epoch_val_accuracies) / len(epoch_val_accuracies)

            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_accuracy)

        print(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {avg_train_loss:.4f}, Train accuracy: {avg_train_accuracy * 100:.2f}%, Val loss: {avg_val_loss:.4f}, Val accuracy: {avg_val_accuracy * 100:.2f}%")

        lr_scheduler.step()

        # Save model every 40 epochs (adjust this if needed)
        if (epoch + 1) % 1 == 0:
            save_path = os.path.join(save_dir, f"linear_probe_epoch_{epoch + 1}.pth")
            torch.save(linear_probe.state_dict(), save_path)
            print(f"Model checkpoint saved at: {save_path}")

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
