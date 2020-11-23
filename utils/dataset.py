import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split


def split_data(data, test_proportion, valid_proportion):
    """
    Split a Pytorch ImageFolder into train/test/valid

    Parameters
    ----------
    data: torchvision ImageFolder
        ImageFolder object with all images to be used
    test_proportion, valid_proportion: float in [0,1]
        Respectively test and valid proportions to use in our dataset

    Returns
    -------
    data_train, data_test, data_valid: torch.utils.data.dataset.Subset
        Respectively train/test/valid subset of the full dataset
    """
    n_images = len(data)
    n_test = int(test_proportion * n_images)
    n_valid = int(valid_proportion * n_images)
    n_train = n_images - (n_valid + n_test)
    lengths = [n_train, n_test, n_valid]
    data_train, data_test, data_valid = random_split(data, lengths)
    return data_train, data_test, data_valid


def get_test_transform(im_size):
    """Returns transformation for test images (resize, convert to tensor and normalize)"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    tfm_test = transforms.Compose([transforms.Resize((im_size, im_size)),
                                   transforms.ToTensor(),
                                   normalize
                                   ])
    return tfm_test


def set_test_transform(data, im_size):
    """Set transformation to test or valid splitted ImageFolder"""
    tfm_test = get_test_transform(im_size)
    data.dataset.transform = tfm_test
    return data


def get_train_transform(im_size, prob_hflip, rotation_degree):
    """Returns train images transformation with data augmentation"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    tfm_train = transforms.Compose([transforms.Resize((im_size, im_size)),
                                    transforms.RandomHorizontalFlip(prob_hflip),
                                    transforms.RandomRotation(rotation_degree),
                                    transforms.ToTensor(),
                                    normalize
                                    ])
    return tfm_train


def set_train_transform(data, im_size, prob_hflip, rotation_degree):
    """Set transformation to train splitted ImageFolder"""
    tfm_train = get_train_transform(im_size, prob_hflip, rotation_degree)
    data.dataset.transform = tfm_train
    return data


def un_normalize(images, device):
    """Un-normalize a batch of RGB images"""
    mean = torch.as_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = torch.as_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    images = (images * std) + mean
    return images


def get_dataloaders(data_path, batch_size,
                    test_proportion, valid_proportion,
                    prob_hflip, im_size, rotation_degree):
    """
    Returns Pytorch DataLoaders for train/test/valid set

    Parameters
    ----------
    data_path: str
        Data path with one folder for one class of images
    batch_size: int
        Batch size for the dataloaders
    test_proportion, valid_proportion: float in [0,1]
        Respectively test and validation proportion size
    im_size: int
        Size of the image (im_size x im_size)
    prob_hflip: float in [0,1]
        Probability to do data augmentation by horizontally flipping image
    rotation_degree: int
        Degree of rotation for data augmentation

    Returns
    -------
    train_data_loader, test_data_loader, valid_data_loader: Pytorch DataLoader
        Respectively train/test/valid dataloaders
    """
    data = ImageFolder(root=data_path)
    data_train, data_test, data_valid = split_data(data, test_proportion, valid_proportion)
    data_train = set_train_transform(data_train, im_size, prob_hflip, rotation_degree)
    data_test = set_test_transform(data_test, im_size)
    data_valid = set_test_transform(data_valid, im_size)
    train_data_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    valid_data_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader, valid_data_loader


def sample_data(data_loader, n_to_sample):
    """
    Sample data from a Pytorch DataLoader

    Parameters
    ----------
    data_loader: torch DataLoader
        Data loader to sample from
    n_to_sample: int
        Number of samples to return

    Returns
    -------
    data: torch.Tensor
        Tensor of all images
    labels: torch.Tensor
        Labels associated to images in data
    """
    x, y = next(iter(data_loader))
    data = x[:n_to_sample]
    labels = y[:n_to_sample]
    return data, labels
