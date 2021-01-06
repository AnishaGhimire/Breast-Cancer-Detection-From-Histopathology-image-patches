import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def preprocess():
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    batch_size = 32
    num_workers = 0

    train_transform = transforms.Compose([transforms.Resize(50),
                                      transforms.CenterCrop(50),
                                      transforms.RandomVerticalFlip(p=0.3),
                                      transforms.RandomHorizontalFlip(p=0.3),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means, stds)])

    transform = transforms.Compose([transforms.Resize(50),
                                transforms.CenterCrop(50),
                                transforms.ToTensor(),
                                transforms.Normalize(means, stds)])

    train_dataset = datasets.ImageFolder(
        root='IDC_Balanced_dataset', transform=train_transform)
    valid_dataset = datasets.ImageFolder(
        root='IDC_Balanced_dataset', transform=transform)

    train_size = 0.8
    num_train = len(train_dataset)
    indices = list(range(num_train))

    valid_split = int(np.floor(train_size * num_train))
    test_split = int(np.floor((train_size+(1-train_size)/2) * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx, test_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    loaders = {
        'train': torch.utils.data.DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         sampler=train_sampler,
                                         num_workers=num_workers),

        'valid': torch.utils.data.DataLoader(valid_dataset,
                                         batch_size=batch_size,
                                         sampler=valid_sampler,
                                         num_workers=num_workers),

        'test': torch.utils.data.DataLoader(valid_dataset,
                                        batch_size=batch_size,
                                        sampler=test_sampler,
                                        num_workers=num_workers)
    }

    return loaders
