import torch
from torchvision import transforms
import json
from PIL import Image


### ---  Predict utility functions  ---###
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = checkpoint['model']
    arch = checkpoint['arch']
    
    if arch in 'vgg' or arch in 'densenet':
        model.classifier = checkpoint['classifier']
    elif arch in 'resnet':
        model.fc = checkpoint['classifier']
    elif arch in 'vit':
        model.heads.head = checkpoint['classifier']

    # model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])  # Call the method
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    model.epochs = checkpoint['epochs']
#     model.criterion = checkpoint['criterion']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # scale, crop and normalize pil image as manner as trained using transforms.Compose()
    pil_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # open image and apply pil_transforms
    pil_image = Image.open(image)
    pil_image = pil_transform(pil_image)

    return pil_image


def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.to(device)
    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.float()

    if device == 'cuda':
        with torch.no_grad():
            output = model.forward(image.cuda())
    else:
        model.to('cpu')
        with torch.no_grad():
            output = model.forward(image)

    probability = torch.exp(output)

    # highest k prob tensors and their indices
    probs, indices = probability.topk(topk)

    probs = probs.cpu()
    indices = indices.cpu()

    # dict inverse
    invert_map = {index: itm for itm, index in model.class_to_idx.items()}

    classes = []
    for index in indices.numpy()[0]:
        classes.append(invert_map[index])

    return probs.numpy()[0], classes


def map_category(file, classes):
    class_list = []

    with open(file, 'r') as f:
        cat_to_names = json.load(f)

    for cls in classes:
        class_list.append(cat_to_names[cls])

    return class_list
