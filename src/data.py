"""
Data loader for PathMNIST with simple augmentations.
"""
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import PathMNIST

def get_dataloaders(root="./data", batch_size=128, workers=2, rgb=True):
    if rgb:
        tfm = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
        in_ch = 3
    else:
        tfm = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5]),
        ])
        in_ch = 1
    trainset = PathMNIST(root=root, split='train', transform=tfm, download=True)
    testset  = PathMNIST(root=root, split='test',  transform=tfm, download=True)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True),
        DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        in_ch
    )
