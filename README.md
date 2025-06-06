# Flower Image Classifier

This project implements a deep learning model to classify images of flowers. It includes scripts for training a model, predicting flower types from images, and a web interface using Gradio.

## Project Structure

-   `app.py`: Launches a Gradio web interface for image classification.
-   `predict.py`: Script for command-line prediction using a trained model.
-   `train.py`: Script for training the deep learning model.
-   `requirements.txt`: Lists the necessary Python dependencies.
-   `cat_to_name.json`: Maps category indices to flower names (used by `app.py` and `predict.py`).
-   `checkpoints/`: Directory to store trained model checkpoints.
-   `flowers/`: Directory containing the dataset (expected structure: `train/`, `valid/`, `test/` subdirectories, each with class subdirectories).
-   `utils/`: Contains utility scripts for data preprocessing and prediction.

## Setup

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  Install the required Python packages. It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The project expects the dataset to be organized in the `flowers/` directory with `train/`, `valid/`, and `test/` subdirectories. Each of these should contain subdirectories named after the class index (e.g., `1/`, `2/`, ...), holding the images for that class.

## Training

To train a new model, run the `train.py` script. You can specify various parameters like architecture, learning rate, epochs, etc.

```bash
python train.py --data_dir flowers --save_dir checkpoints --arch resnet50 --epochs 10 --learning_rate 0.001
```

This will train a ResNet50 model on the flowers dataset and save the checkpoint in the checkpoints directory.

## Prediction (Command Line)
Use the `predict.py` script to classify an image from the command line.

```
python predict.py /path/to/image.jpg checkpoints/resnet50_model_checkpoint_10ep.pth --category_names cat_to_name.json --top_k 5
```
Replace `/path/to/image.jpg` with the path to your image and `checkpoints/resnet50_model_checkpoint_10ep.pth` with the path to your trained model checkpoint.

## Gradio Web Interface
A simple web interface is provided using Gradio for easy image classification.

1. Ensure you have a trained model checkpoint (e.g., `vit_b_16_model_checkpoint_7ep.pt` as configured in `app.py`) and the `cat_to_name.json` file. Update the `MODEL_PATH` and `CATEGORY_NAMES_PATH` variables in `app.py` if necessary.

2. Run the `app.py` script: `python app.py`

3. The script will print a local URL (e.g., `http://127.0.0.1:7860`) where the Gradio interface is running. Open this URL in your web browser.

### Using the Gradio Interface
- Open the URL provided after running python app.py.
- You will see a simple interface with an image upload area and a predictions output area.
- Click on the image upload area or drag and drop an image of a flower.
- The model will process the image, and the top predicted flower names and their probabilities will appear in the predictions area.

## License
This project is licensed under the MIT License - see the LICENSE file for details.