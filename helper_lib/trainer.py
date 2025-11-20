import torch
from helper_lib.model import diffusion_schedule
# Assignment 2
def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=5):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    print("✅ Training completed!")
    return model

# ===  GAN TRAINING FUNCTION (Assignment3) ===
def train_gan(models, data_loader, device='cpu', epochs=10, lr=0.0002):
    import torch
    import torch.nn as nn

    G, D = models["generator"].to(device), models["discriminator"].to(device)
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            batch_size = imgs.size(0)

            # --- Train Discriminator ---
            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = G(z)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            D_real = D(imgs)
            D_fake = D(fake_imgs.detach())
            loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Train Generator ---
            D_fake = D(fake_imgs)
            loss_G = criterion(D_fake, real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}")

    return G, D

# ================================================================
# DIFFUSION TRAINING FUNCTION (Activity 6 / Assignment 4)
# ================================================================
def train_diffusion(model, data_loader, optimizer, device='cpu', epochs=10):
    """
    Trains a DiffusionModel using the forward-diffusion objective:
        model predicts the added noise (epsilon) for random timesteps t.

    model: DiffusionModel (has unet + time embedding + EMA)
    data_loader: image loader (images must be in [-1,1] or [0,1])
    optimizer: optimizer for model.unet parameters
    device: 'cpu' or 'cuda'
    epochs: number of training epochs
    """
    import torch.nn.functional as F
    model.to(device)

    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, _ in data_loader:
            images = images.to(device)

            # Normalize input to [-1, 1]
            images = images * 2 - 1  # assuming original in [0,1]

            batch_size = images.size(0)

            # Sample random diffusion times t ~ Uniform(0,1)
            t = torch.rand(batch_size, 1, device=device)

            # Compute noise rates and signal rates
            noise_rates, signal_rates = diffusion_schedule(t)

            # Reshape for broadcasting
            noise_rates = noise_rates.view(batch_size, 1, 1, 1)
            signal_rates = signal_rates.view(batch_size, 1, 1, 1)

            # Sample random Gaussian noise
            noise = torch.randn_like(images)

            # Apply forward diffusion: x_t = s * x_0 + n * noise
            noisy_images = signal_rates * images + noise_rates * noise

            # Time embedding for U-Net
            t_emb = model.time_emb(t)  # (B, 64)

            # Predict noise from noisy image
            pred_noise = model.unet(noisy_images, t_emb)

            # Loss: predict exact noise
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA network for sampling
            model.update_ema()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")

    print("✅ Diffusion training completed!")
    return model

# ================================================================
# Energy-Based Model Training (EBM) - Assignment 4
# ================================================================
def train_ebm(energy_model, data_loader, device="cpu",
              epochs=5, langevin_steps=60, step_size=0.1, noise_std=0.01):
    """
    Trains an Energy-Based Model using a simple contrastive divergence objective:
        loss = E_data[ E(x_data) ] - E_model[ E(x_model) ]

    - Positive samples: real images from CIFAR-10.
    - Negative samples: generated by Langevin dynamics starting from random noise.
    """
    energy_model.to(device)
    optimizer = torch.optim.Adam(energy_model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        energy_model.train()
        running_loss = 0.0
        print(f"[EBM] Start epoch {epoch+1}/{epochs}")

        for images, _ in data_loader:
            images = images.to(device)

            # Assume input in [0,1], map to [-1,1] for stability
            images = images * 2 - 1

            batch_size = images.size(0)

            # --- Initialize negative samples from random noise in [-1,1] ---
            x_neg = torch.rand_like(images) * 2 - 1
            x_neg = x_neg.to(device)

            # --- Langevin Dynamics to move x_neg towards low-energy regions ---
            x_neg = x_neg.detach()
            for _ in range(langevin_steps):
                x_neg.requires_grad_(True)
                energy_neg = energy_model(x_neg).sum()
                grad = torch.autograd.grad(energy_neg, x_neg)[0]

                # Move *against* the gradient (toward lower energy)
                x_neg = x_neg - step_size * grad
                x_neg = x_neg + noise_std * torch.randn_like(x_neg)
                x_neg = x_neg.clamp(-1, 1).detach()

            # --- Compute energies for real (positive) and model (negative) samples ---
            energy_pos = energy_model(images)      # (B,1)
            energy_neg = energy_model(x_neg)       # (B,1)

            # Contrastive divergence style loss
            loss = energy_pos.mean() - energy_neg.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"[EBM] Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")

    print("✅ EBM training completed!")
    return energy_model