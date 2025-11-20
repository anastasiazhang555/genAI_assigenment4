import torch

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")

def load_model(model, path="model.pth", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"✅ Model loaded from {path}")
    return model
