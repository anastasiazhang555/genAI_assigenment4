import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),              
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        transform=transform,
        download=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0
    )
    return loader