import math

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def generate_images(model, device='cpu', num_samples=10, seed=None):
    """Sample num_samples latent vectors and return generated images in [0, 1], shape (N, 1, 28, 28)."""
    generator = model.generator if hasattr(model, "generator") else model
    generator.to(device)
    generator.eval()
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        noise = torch.randn(num_samples, generator.z_dim, device=device)
        images = generator(noise).cpu()
    return (images + 1) / 2  # undo the [-1, 1] normalization (Tanh output)

def generate_samples(model, device='cpu', num_samples=10, seed=None):
    """Generate num_samples points in the latent space, run the generator, and plot the samples on a grid."""
    images = generate_images(model, device, num_samples, seed)
    nrow = math.ceil(math.sqrt(num_samples))
    grid = make_grid(images, nrow=nrow, padding=2, pad_value=1)

    plt.figure(figsize=(nrow * 1.2, math.ceil(num_samples / nrow) * 1.2))
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    plt.title(f"{num_samples} generated digits")
    plt.axis("off")
    plt.show()
