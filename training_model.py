import torch 
from utils import setup_seed ,  count_parameters  , loading_data ,  ImageDataset
from modeling_mae import MAE 
from configuration import Config 
from  torch.utils.data import DataLoader 
from tqdm.auto import tqdm 
import time 

config = Config()
setup_seed(seed=42)

COMPILE = True 
EPOCHS = 10 
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_INTERVAL  = 100 

train_data , test_data = loading_data(dataset_name="ethz/food101",num_data=40_000)
train_dataset = ImageDataset(train_data)
test_dataset = ImageDataset(test_data)

train_loader=  DataLoader(
    train_dataset,
    batch_size = 32, 
    shuffle=True,
    num_workers=4
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

model = Mae(config)
print("Model Parameters:",count_parameters(model))

if COMPILE:
    model = torch.compile(model)
    torch.set_float32_matmul_precision('high')

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

model.to(DEVICE)
print("Starting to train")

# Initialize lists to store metrics
train_loss = []
val_loss = []
lr = []
step_times = []
grad_norms = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, image in progress_bar:
        start_time = time.time()
        image = image.to(DEVICE)

        # Zero the gradients before backpropagation
        optimizer.zero_grad()
        
        # Forward pass
        out, mask, ids_restore = model(image)
        
        
        # Reshape image to match the output shape
        image_patches = image.view(image.size(0), -1, out.size(-1))

        # Apply mask to both out and image_patches
        masked_out = out[mask == 1]
        masked_image_patches = image_patches[mask == 1]

        # Calculate loss only on masked patches
        loss = torch.mean((masked_out - masked_image_patches) ** 2) / 0.75

        # Backward pass and gradient clipping
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update model parameters
        optimizer.step()

        # Track training metrics
        train_loss.append(loss.item())
        grad_norms.append(grad_norm)
        
        end_time = time.time()
        step_times.append((end_time - start_time) * 1000)  # Record step time in milliseconds
        lr.append(scheduler.get_last_lr()[0])

        # Update progress bar with current training metrics
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'grad_norm': f"{grad_norm:.4f}",
            'step_time': f"{step_times[-1]:.4f}ms",
            'lr': f"{lr[-1]:.6f}"
        })

        # Validation phase at specified intervals
        if (step + 1) % EVAL_INTERVAL == 0:
            model.eval()  # Set model to evaluation mode
            val_running_loss = 0.0
            
            with torch.no_grad():  # Disable gradient computation
                progress_bar_eval = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Evaluating")
                
                for val_step, val_image in progress_bar_eval:
                    val_image = val_image.to(DEVICE)
                    
                    # Forward pass on validation images
                    val_out, mask, ids_restore = model(val_image)
                    
                    # Reshape validation image to match the output shape
                    val_image_patches = val_image.view(val_image.size(0), -1, val_out.size(-1))
                    
                    # Apply mask to both val_out and val_image_patches
                    masked_val_out = val_out[mask == 1]
                    masked_val_image_patches = val_image_patches[mask == 1]
                    
                    # Compute validation loss
                    val_loss_value = torch.mean((masked_val_out - masked_val_image_patches) ** 2) / 0.75
                    val_running_loss += val_loss_value.item()

            # Calculate average validation loss and track it
            avg_val_loss = val_running_loss / len(test_loader)
            val_loss.append(avg_val_loss)

            # Update validation progress bar with current metrics
            progress_bar_eval.set_postfix({
                'val_loss': f"{avg_val_loss:.4f}"
            })
            
            model.train()  # Switch back to training mode

    # Step the learning rate scheduler
    scheduler.step()

print("Training complete")


print("Training complete")