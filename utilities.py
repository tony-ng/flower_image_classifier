import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

def load_datasets(data_dir):
    """
    Load datasets for model training.
    Parameters:
    data_dir - the directory of the datasets. There should be two sub-folders, 'train' and 'valid', in this folder for training and validation purposes
    Returns:
    train_dataset - the datasets for training purpose
    valid_dataset - the datasets for validation purpose
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    return train_dataset, valid_dataset

def load_dataloader(dataset, for_training = False):
    """
    Create dataloader for the dataset
    Parameters:
    dataset - the dataset to get the dataloader
    for_training - the dataset is used for training
    Returns:
    dataloader - the dataloader for the input dataset
    """
    shuffle = False
    batch = 32
    if for_training:
        shuffle = True
        batch = 64
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle)
    return dataloader

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    resize_size = 256
    crop_size = 224
    ori_width, ori_height = im.size
    
    width = 0
    height = 0
    if ori_height > ori_width:
        height = int(ori_height * resize_size / ori_width)
        width = resize_size
    else:
        width = int(ori_width * resize_size / ori_height)
        height = resize_size    
    im.thumbnail((width, height))
    
    left = (width - crop_size)/2
    top = (height - crop_size)/2
    right = (width + crop_size)/2
    bottom = (height + crop_size)/2
    
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im) / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    return np_image.transpose((2, 0, 1))
