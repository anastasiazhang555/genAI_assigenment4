import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# ================================================================
# 1. CNN Classifier
# ================================================================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ================================================================
# 2. GAN Models
# ================================================================
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ================================================================
# 3. Diffusion Model Components (Activity 6)
# ================================================================

# ---- Sinusoidal Time Embedding ----
class SinusoidalEmbedding(nn.Module):
    """
    Time embedding to encode diffusion timestep t.
    """
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.weights = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), dim))

    def forward(self, t):
        """
        t: (B, 1) in [0,1]
        return: (B, 2*dim)
        """
        t = t * self.weights.to(t.device)
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


# ---- Simple U-Net for Denoising ----
class SimpleUNet(nn.Module):
    """
    A lightweight UNet suitable for 64x64 diffusion.
    """
    def __init__(self, input_channels=3):
        super().__init__()

        self.down1 = nn.Conv2d(input_channels, 32, 4, 2, 1)  # 64→32
        self.down2 = nn.Conv2d(32, 64, 4, 2, 1)              # 32→16
        self.down3 = nn.Conv2d(64, 128, 4, 2, 1)             # 16→8

        self.mid = nn.Conv2d(128, 128, 3, 1, 1)

        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(32, input_channels, 4, 2, 1)

        self.time_embed = nn.Linear(32 * 2, 128)

    def forward(self, x, t_emb):
        # down
        d1 = F.relu(self.down1(x))
        d2 = F.relu(self.down2(d1))
        d3 = F.relu(self.down3(d2))

        # time embedding
        t = F.relu(self.time_embed(t_emb))
        t = t[..., None, None]  # (B,128,1,1)

        # mid
        m = F.relu(self.mid(d3 + t))

        # up
        u1 = F.relu(self.up1(m))
        u2 = F.relu(self.up2(u1))
        u3 = self.up3(u2)
        return u3


# ---- Noise schedule ----
def diffusion_schedule(t, min_rate=0.02, max_rate=0.9):
    """
    Maps t in [0,1] → (noise_rate, signal_rate)
    """
    start = math.acos(max_rate)
    end = math.acos(min_rate)
    angle = start + t * (end - start)
    return torch.sin(angle), torch.cos(angle)


# ---- Full Diffusion Wrapper ----
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = SimpleUNet(3)
        self.time_emb = SinusoidalEmbedding(dim=32)

        # EMA network: used for more stable sampling
        self.ema = copy.deepcopy(self.unet)
        self.ema_decay = 0.9

    def schedule_fn(self, t):
        return diffusion_schedule(t)

    def update_ema(self):
        """
        Update EMA parameters: ema = decay * ema + (1 - decay) * current.
        """
        with torch.no_grad():
            for p, ema_p in zip(self.unet.parameters(), self.ema.parameters()):
                ema_p.data = self.ema_decay * ema_p.data + (1 - self.ema_decay) * p.data

# ================================================================
# 4. Energy-Based Model (EBM) for CIFAR-10
# ================================================================
class EnergyModel(nn.Module):
    """
    A simple CNN-based Energy Model.
    It takes an image input and outputs a scalar "energy".
    Lower energy = more likely under the model.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),
        )
        # assuming 64x64 input → after 3 downsamples → 8x8
        self.fc = nn.Linear(128 * 8 * 8, 1)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        energy = self.fc(h)
        return energy  # (B, 1)

# ================================================================
# 5. Unified get_model()
# ================================================================
def get_model(model_name="CNN"):
    model_name = model_name.upper()

    if model_name == "CNN":
        return CNNClassifier(num_classes=10)

    elif model_name == "GAN":
        return {"generator": Generator(), "discriminator": Discriminator()}

    elif model_name == "DIFFUSION":
        return DiffusionModel()

    elif model_name == "EBM":
        # Energy-Based Model for CIFAR-10-like images (3 x 64 x 64)
        return EnergyModel(in_channels=3)

    else:
        raise ValueError(f"Unknown model name: {model_name}")