import torch
from torchvision import datasets, transforms


### ---  Train utility functions  ---###
def process_data(dir_path, batch):
    '''
    Arguments: the path to image data
    Returns: The loaders for train, validation and test datasets

    This function receives the directory path of the image data and apply necessary transformations

    '''

    train_dir = dir_path + '/train'
    valid_dir = dir_path + '/valid'
    test_dir = dir_path + '/test'

    # Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(100),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(
        train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(
        valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch, shuffle=True)
    validloader = torch.utils.data.DataLoader(
        valid_datasets, batch_size=batch, shuffle=False)

    print('<---- Loading {} directory into the network ---->'.format(dir_path))

    return trainloader, testloader, validloader, train_datasets
