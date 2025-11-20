"""
Assignment 4 — Model Deployment API
Includes:
    - CNN classification (CIFAR-10)
    - GAN training & generation (MNIST)
    - Diffusion model training & generation (CIFAR-10)
    - Energy-Based Model (EBM) training & generation (CIFAR-10)

"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from PIL import Image
import io
import os

# helper library imports
from helper_lib.model import get_model
from helper_lib.trainer import train_model, train_gan, train_diffusion, train_ebm
from helper_lib.generator import generate_gan_samples, generate_diffusion_samples, generate_ebm_samples
from helper_lib.data_loader import get_data_loader


# =======================================================
# FastAPI Initialization
# =======================================================
app = FastAPI(title="Assignment 4 API", version="3.0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================================================
# 1 — CNN Initialization
# =======================================================
cnn_model = get_model("CNN").to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

try:
    cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
    print("Loaded pretrained CNN weights.")
except:
    print("Training new CNN model...")
    train_loader = get_data_loader("./data", batch_size=64, train=True)
    cnn_model = train_model(cnn_model, train_loader, criterion, optimizer, device=device, epochs=5)
    torch.save(cnn_model.state_dict(), "cnn_model.pth")


# =======================================================
# 2 — GAN Initialization
# =======================================================
gan_dict = get_model("GAN")
G = gan_dict["generator"].to(device)
D = gan_dict["discriminator"].to(device)


# =======================================================
# API ROUTES
# =======================================================

@app.get("/")
def home():
    return {"message": "Assignment 4 API is running."}


# -------------------------------------------------------
# 1. CNN Prediction
# -------------------------------------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict a CIFAR-10 class using the CNN classifier.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(x)
        _, pred = torch.max(outputs, 1)

    class_names = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

    return {"predicted_class_id": pred.item(),
            "label": class_names[pred.item()]}


# -------------------------------------------------------
# 2. GAN Training
# -------------------------------------------------------
@app.get("/train_gan")
def api_train_gan(epochs: int = 3):
    mnist = datasets.MNIST("./data", train=True, download=True,
                           transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)

    trained_G, trained_D = train_gan({"generator": G, "discriminator": D},
                                     loader, device=device, epochs=epochs)

    torch.save(trained_G.state_dict(), "gan_generator.pth")
    torch.save(trained_D.state_dict(), "gan_discriminator.pth")
    return {"message": f"GAN trained for {epochs} epochs."}


# -------------------------------------------------------
# 3. GAN Generation
# -------------------------------------------------------
@app.get("/generate_gan")
def api_generate_gan(num_samples: int = 5):
    try:
        G.load_state_dict(torch.load("gan_generator.pth", map_location=device))
    except:
        pass  # If not found, use untrained generator

    path = generate_gan_samples(G, device=device, num_samples=num_samples)
    return {"message": "GAN images generated.", "file_path": path}


# -------------------------------------------------------
# 4. Diffusion Training (CIFAR-10)
# -------------------------------------------------------
@app.get("/train_diffusion")
def api_train_diffusion(epochs: int = 1):  
    diffusion_model = get_model("DIFFUSION").to(device)

    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    cifar = datasets.CIFAR10("./data", train=True, download=True, transform=transform)

    subset = torch.utils.data.Subset(cifar, range(2000))

    loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)

    optimizer = optim.Adam(diffusion_model.unet.parameters(), lr=1e-4)

    trained = train_diffusion(diffusion_model, loader, optimizer,
                              device=device, epochs=epochs)

    torch.save(trained.state_dict(), "diffusion_model.pth")
    return {"message": f"Diffusion model trained for {epochs} epoch(s) on 2000 samples."}


# -------------------------------------------------------
# 5. Diffusion Image Generation
# -------------------------------------------------------
@app.get("/generate_diffusion")
def api_generate_diffusion(num_samples: int = 6):
    diffusion_model = get_model("DIFFUSION").to(device)

    try:
        diffusion_model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))
    except:
        pass

    path = generate_diffusion_samples(diffusion_model, device=device,
                                      num_samples=num_samples)
    return {"message": "Diffusion images generated.", "file_path": path}


# -------------------------------------------------------
# 6. EBM Training
# -------------------------------------------------------
@app.get("/train_ebm")
def api_train_ebm(epochs: int = 1):
    ebm = get_model("EBM").to(device)

    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    cifar = datasets.CIFAR10("./data", train=True, download=True, transform=transform)

    subset = torch.utils.data.Subset(cifar, range(1000))
    loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)

    trained = train_ebm(ebm, loader, device=device,
                        epochs=epochs, langevin_steps=20, step_size=0.05, noise_std=0.01)

    torch.save(trained.state_dict(), "ebm_model.pth")
    return {"message": f"EBM trained for {epochs} epoch(s) on 1000 samples."}


# -------------------------------------------------------
# 7. EBM Generation
# -------------------------------------------------------
@app.get("/generate_ebm")
def api_generate_ebm(num_samples: int = 6):
    ebm = get_model("EBM").to(device)

    try:
        ebm.load_state_dict(torch.load("ebm_model.pth", map_location=device))
    except:
        pass

    path = generate_ebm_samples(ebm, device=device, num_samples=num_samples)
    return {"message": "EBM images generated.", "file_path": path}