import torch
import time
import torchvision
from tqdm.auto import tqdm
from datasets import load_dataset
from modeling_mae import MAE_ViT
from inference import run_inference
from torchvision import transforms
import torch.nn.functional as F
from configuration import MAEConfig
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from utils import setup_seed, count_parameters, loading_data, ImageDataset
import argparse
import math
import os
from PIL import Image


# Setting up argparse for CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train MAE_ViT on CIFAR-10")
    parser.add_argument('--epochs', type=int, default=120, help="Number of training epochs (default: 120)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training and validation (default: 128)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for optimizer (default: 1e-4)")
    parser.add_argument('--eval_interval', type=int, default=100, help="Evaluation interval during training (default: 100 steps)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument('--image_path', type=str, help="Path to an image for inference visualization")
    parser.add_argument('--mask_ratio', type=float, default=0.75, help="Masking ratio for MAE (default: 0.75)")

    return parser.parse_args()


def save_metrics(epoch, train_losses, val_losses, step_times, learning_rate):
    with open(f'metrics_epoch_{epoch}.txt', 'w') as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Val Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Step Times (ms): {step_times}\n")
        f.write(f"Learning Rate: {learning_rate:.6f}\n")


def main():
    args = parse_args()

    # Directory for saving model checkpoints
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    config = MAEConfig()

    setup_seed(seed=args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)

    transform = Compose([
        Resize((32, 32)),  
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  
    ])

    # Load the CIFAR-10 dataset with resizing
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = MAE_ViT(config=config)
    print("Model Parameters:", count_parameters(model))
    model.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    lr_func = lambda epoch: min((epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / args.epochs * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    print("Starting to train")

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    step_times = []

    def calculate_loss(preds, image, mask, mask_ratio):
        return torch.mean((preds - image) ** 2 * mask) / mask_ratio

    for epoch in range(args.epochs):
        model.train()
        epoch_train_losses = []
        step_times = []
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, (image, _) in progress_bar:
            start_time = time.time()
            image = image.to(DEVICE)

            optimizer.zero_grad()
            out, mask = model(image)
            loss = calculate_loss(preds=out, image=image, mask=mask, mask_ratio=args.mask_ratio)

            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            step_times.append((time.time() - start_time) * 1000)  # Step time in milliseconds
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'step_time': f"{(time.time() - start_time) * 1000:.4F}ms",
                'lr': f"{lr_scheduler.get_last_lr()[0]:.6f}"
            })

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)

        # Validation step
        model.eval()  # Set the model to evaluation mode
        epoch_val_losses = []

        with torch.no_grad():  # Disable gradient tracking
            val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation {epoch+1}/{args.epochs}")
            
            for val_step, (val_image, _) in val_progress_bar:
                val_image = val_image.to(DEVICE)
                out, mask = model(val_image)
                val_loss = calculate_loss(preds=out, image=val_image, mask=mask, mask_ratio=args.mask_ratio)
                epoch_val_losses.append(val_loss.item())

            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")

        lr_scheduler.step()

        # Save model and metrics every 40 epochs
        if (epoch + 1) % 40 == 0:
            save_path = os.path.join(save_dir, f"mae_vit_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved at: {save_path}")
            
            # Save metrics to text files
            save_metrics(epoch + 1, train_losses, val_losses, step_times, lr_scheduler.get_last_lr()[0])

if __name__ == "__main__":
    main()