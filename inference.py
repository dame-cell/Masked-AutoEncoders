import torch
import numpy as np
import matplotlib.pyplot as plt

# Adjust normalization constants for the range [0, 1] instead of ImageNet specifics
image_mean = np.array([0.5, 0.5, 0.5])
image_std = np.array([0.5, 0.5, 0.5])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    
    # Denormalize the image
    denormalized = image * image_std + image_mean
    
    # Clip values to [0, 1] range and convert to uint8
    img_display = torch.clip(denormalized, 0, 1).numpy() * 255
    img_display = img_display.astype(np.uint8)
    
    plt.imshow(img_display)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def run_inference(image, model):
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to the model's input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=image_mean, std=image_std)  # Normalize
    ])
    model.eval()  # Ensure the model is in evaluation mode
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    model.to("cpu")
    image = image.to("cpu")
    
    with torch.no_grad():
        out, mask = model(image)
        
        # Both out and mask are already in the correct shape [1, 3, 32, 32]
        y = out.detach().cpu()
        mask = mask.detach().cpu()
        # Convert to NHWC format for visualization
        x = image.permute(0, 2, 3, 1).cpu()  # Original image
        y = y.permute(0, 2, 3, 1)  # Reconstruction
        mask = mask.permute(0, 2, 3, 1)  # Mask
        # Masked image
        im_masked = x * (1 - mask)
        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask
        
        # Plot results
        plt.rcParams['figure.figsize'] = [24, 24]
        plt.subplot(1, 4, 1)
        show_image(x[0], "Original")
        plt.subplot(1, 4, 2)
        show_image(im_masked[0], "Masked")
        plt.subplot(1, 4, 3)
        show_image(y[0], "Reconstruction")
        plt.subplot(1, 4, 4)
        show_image(im_paste[0], "Reconstruction + Visible")
        plt.show()
