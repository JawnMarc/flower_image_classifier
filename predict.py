import torch
import argparse
from utils.predict_utils import predict, load_checkpoint, map_category

# argument object
parser = argparse.ArgumentParser()
parser.add_argument('input_image', default='flower/test/77/image_00005.jpg',
                    action='store', help='Specify image location')
parser.add_argument('checkpoint', default='checkpoint.pth',
                    help='Specify checkpoint')
parser.add_argument('--top_k', type=int, default=5,
                    help='Specify top K most likely classes')
parser.add_argument('--category_names',
                    help='Specify file for the category names')
parser.add_argument('--gpu', action='store_true',
                    help='Specify the use of gpu power over cpu')


# paarsing arguments
args = parser.parse_args()


input_image = args.input_image
checkpoint_path = args.checkpoint  # Renamed for clarity
top_k = args.top_k
category_names_file = args.category_names  # Use argument directly
gpu = args.gpu
device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
# device = 'cuda' if gpu and torch.backends.mps.is_available() else 'cpu'

# Load train model
model = load_checkpoint(checkpoint_path)

# Probability class
probs, class_ids = predict(input_image, model, top_k, device)

print('Image: ', input_image)
print('Probabilities / Classes')

if category_names_file:
    class_labels = map_category(category_names_file, class_ids)
else:
    class_labels = class_ids  # Fallback to IDs if no mapping file

# print names of classes
for label, prob in zip(class_labels, probs):
    print('{:.4f} / {} '.format(prob, label))