import torch
from torchvision import datasets, transforms
from PIL import Image

def dataloader(TRAIN_SIZE, TEST_SIZE, num_workers, dir, classification, ResizeIMG, use_cuda):
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {'num_workers': num_workers}
    if ResizeIMG:
        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(dir, classification, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Resize((299, 299), interpolation=Image.BICUBIC), # Resize Image to match min requirement of Inception
                            transforms.ToTensor()
                        ])),
            batch_size=TRAIN_SIZE, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(dir, classification, train=False, transform=transforms.Compose([
                            transforms.Resize((299, 299), interpolation=Image.BICUBIC),
                            transforms.ToTensor()
                        ])),
            batch_size=TEST_SIZE, shuffle=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(dir, classification, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
            batch_size=TRAIN_SIZE, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(dir, classification, train=False, transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
            batch_size=TEST_SIZE, shuffle=True, **kwargs)
    return train_loader, test_loader