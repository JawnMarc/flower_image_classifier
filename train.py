import torch
import argparse

from utils.train_utils import train_model, save_checkpoint
from utils.data_preprocess import process_data
from models.model_config import model_setup


# argument object
parser = argparse.ArgumentParser()
# adding arguments
parser.add_argument('data_dir', action='store',
                    help='Specify the image data directory, default is flowers/')
parser.add_argument('--save_dir', default='checkpoints',
                    help='Specify directory to save file')
parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'vgg19', 'densenet121', 'resnet18', 'resnet50',
                                                        'vit_b_16', 'vit_l_16'], help='Specify the model architecture')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Specify the learning rate for your model training')
parser.add_argument('--hidden_layers', type=int, default=512,
                    help='Specify the hidden units of your model')
parser.add_argument('--epochs', type=int, default=5,
                    help='Specify the number of epochs')
parser.add_argument('--gpu', action='store_true',
                    help='Specify the use of gpu power over cpu')
parser.add_argument('--batch', type=int, default=32,
                    help='Specify the batch size per epoch')

# paarsing arguments
args = parser.parse_args()

dir_path = args.data_dir
save_dir = args.save_dir
lr = args.learning_rate
architect = args.arch
hidden_layer = args.hidden_layers
epochs = args.epochs
gpu = args.gpu
batch_size = args.batch

# device agnostic to detect gpu or cpu
device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
# device = 'cuda' if gpu and torch.backends.mps.is_available() else 'cpu'

# Load datasets and apply transformations
trainloader, testloader, validloader, train_datasets = process_data(
    dir_path, batch_size)

# Model setup
model, classifier, criterion, optimizer = model_setup(
    architect, hidden_layer, lr)

# Network trains
train_model(model=model, trainloader=trainloader,
             validloader=validloader, criterion=criterion,
             optimizer=optimizer, device=device, arch=architect,
             save_dir=save_dir, classifier=classifier,
             dataset=train_datasets, epochs=epochs)

# Saved trained network
save_checkpoint(model, architect, save_dir,
                classifier, optimizer, train_datasets, epochs)
