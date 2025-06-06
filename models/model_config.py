import torch.nn as nn
import torch.optim as optim
from torchvision import models
from collections import OrderedDict


def model_setup(arch, hidden_units, learning_rate, output_size=102):
    '''
    Arguments: The architecture for the network (e.g., vgg16, resnet50, vit_b_16), 
               the hyperparameters for the network (hidden layer units, and learning rate),
               and the output size for the classifier.

    Returns: The model, criterion and optimizer for training
    '''
    arch_lower = arch.lower()
    print(f'<---- Using the {arch} model architecture ---->')

    if arch_lower == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        input_features = model.classifier[0].in_features
    elif arch_lower == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        input_features = model.classifier[0].in_features
    elif arch_lower == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        input_features = model.classifier.in_features
    elif arch_lower == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        input_features = model.fc.in_features
    elif arch_lower == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        input_features = model.fc.in_features
    elif arch_lower == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        input_features = model.fc.in_features
    elif arch_lower == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        input_features = model.fc.in_features
    elif arch_lower == 'resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        input_features = model.fc.in_features
    elif arch_lower == 'vit_b_16': # Basic ViT
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        input_features = model.heads.head.in_features
    elif arch_lower == 'vit_l_16': # Advanced ViT
        model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        input_features = model.heads.head.in_features
    else:
        raise ValueError(
            f'{arch} Invalid model. Supported models include: vgg16, vgg19, densenet121, '
            f'resnet18, resnet34, resnet50, resnet101, resnet152, vit_b_16, vit_l_16.'
        )

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Defining the new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)), # Added a default dropout rate
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replace the classifier based on model type
    if 'vgg' in arch_lower or 'densenet' in arch_lower:
        model.classifier = classifier
    elif 'resnet' in arch_lower:
        model.fc = classifier
    elif 'vit' in arch_lower:
        model.heads.head = classifier # For Vision Transformers, the head is typically replaced

    criterion = nn.NLLLoss()
    # Ensure only the parameters of the new classifier are passed to the optimizer
    if 'vgg' in arch_lower or 'densenet' in arch_lower:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif 'resnet' in arch_lower:
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    elif 'vit' in arch_lower:
        optimizer = optim.Adam(model.heads.head.parameters(), lr=learning_rate)
    else: # Should not happen if arch validation is correct
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    return model, classifier, criterion, optimizer