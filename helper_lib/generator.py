import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ================================================================
# 1. GAN Image Generation
# ================================================================
def generate_gan_samples(generator, device='cpu', num_samples=10):
    """
    Generates images using a trained GAN generator.
    """
    generator.eval()
    z = torch.randn(num_samples, 100, device=device)

    with torch.no_grad():
        samples = generator(z).cpu()

    samples = (samples + 1) / 2  # rescale from [-1,1] to [0,1]

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')

    output_path = "gan_generated.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"✅ Saved GAN images to {output_path}")
    return output_path


# ================================================================
# 2. Diffusion Sampling (Activity 6 / Assignment 4)
# ================================================================
def generate_diffusion_samples(model, device='cpu', num_samples=8, diffusion_steps=100):
    """
    Generates new images from a trained DiffusionModel using
    the reverse diffusion process.

    model: DiffusionModel with .unet and .ema
    """
    model.eval()
    model.to(device)

    image_size = 64         # matches SimpleUNet input size
    channels = 3            # RGB output

    # Start from pure Gaussian noise
    noise = torch.randn(num_samples, channels, image_size, image_size, device=device)

    # Reverse diffusion process
    step_size = 1.0 / diffusion_steps
    current = noise

    for step in range(diffusion_steps):
        # Compute t and next t
        t = torch.ones(num_samples, 1, device=device) * (1 - step * step_size)
        next_t = t - step_size

        # Map t → noise & signal rates
        noise_rate, signal_rate = model.schedule_fn(t)
        next_noise_rate, next_signal_rate = model.schedule_fn(next_t)

        # Reshape for broadcasting
        noise_rate = noise_rate.view(num_samples, 1, 1, 1)
        signal_rate = signal_rate.view(num_samples, 1, 1, 1)
        next_noise_rate = next_noise_rate.view(num_samples, 1, 1, 1)
        next_signal_rate = next_signal_rate.view(num_samples, 1, 1, 1)

        # Time embedding
        t_emb = model.time_emb(t)

        with torch.no_grad():
            # Predict noise using EMA U-Net for stability
            pred_noise = model.ema(current, t_emb)

            # Predict clean image
            pred_image = (current - noise_rate * pred_noise) / signal_rate

            # Next step
            current = next_signal_rate * pred_image + next_noise_rate * pred_noise

    final_images = current.clamp(-1, 1)
    final_images = (final_images + 1) / 2  # rescale to [0,1]

    # Plot result
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
    for i in range(num_samples):
        img = final_images[i].permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis("off")

    output_path = "diffusion_generated.png"
    plt.savefig(output_path)
    plt.close(fig)

    print(f"✅ Saved Diffusion images to {output_path}")
    return output_path

# ================================================================
# 3.EBM Sampling with Langevin Dynamics
# ================================================================
def generate_ebm_samples(energy_model, device="cpu",
                         num_samples=8, langevin_steps=80, step_size=0.1, noise_std=0.01):
    """
    Generates images from a trained Energy-Based Model using Langevin Dynamics.
    """
    energy_model.to(device)
    energy_model.eval()

    image_size = 64
    channels = 3

    # Start from random noise in [-1,1]
    x = torch.rand(num_samples, channels, image_size, image_size, device=device) * 2 - 1
    x = x.detach()

    for _ in range(langevin_steps):
        x.requires_grad_(True)
        energy = energy_model(x).sum()
        grad = torch.autograd.grad(energy, x)[0]

        x = x - step_size * grad
        x = x + noise_std * torch.randn_like(x)
        x = x.clamp(-1, 1).detach()

    # Map back to [0,1] for visualization
    x_vis = (x + 1) / 2
    x_vis = x_vis.clamp(0, 1).cpu()

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        img = x_vis[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis("off")

    output_path = "ebm_generated.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"✅ Saved EBM images to {output_path}")
    return output_path