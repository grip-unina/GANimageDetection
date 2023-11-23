"""
This module is used for running inference on a single image using pre-trained
models. It supports different models and applies necessary transformations to the
input image before feeding it to the models for prediction.

It prints out the logits returned by each model and the final label based on these logits.
"""

import os
import time
import argparse
import json
import psutil
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from resnet50nodown import resnet50nodown

def compress_and_resize_image(image_path, max_size=(1024, 1024)):
    """
    Compresses and resizes an image to a manageable size.

    Args:
        image_path (str): Path to the image file.
        max_size (tuple): Maximum width and height of the resized image.

    Returns:
        str: Path to the processed image.
    """
    try:
        # Validate the file format
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            raise ValueError("Unsupported file format. Accepts only JPEG, PNG, and WebP.")

        # Open and process the image
        with Image.open(image_path) as img:
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                # Resize the image only if it's larger than max_size
                img.thumbnail(max_size, Image.LANCZOS)
            # Save the processed image in a lossless format
            processed_image_path = os.path.splitext(image_path)[0] + "_processed.png"
            img.save(processed_image_path, format='PNG', optimize=True)
            return processed_image_path

    except UnidentifiedImageError as exc:
        # Explicitly re-raising with context from the original exception
        raise ValueError("Invalid image file or path.") from exc

def print_memory_usage():
    """Prints the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory used: {mem_info.rss / (1024 * 1024):.2f} MB")


models_config = {
    "gandetection_resnet50nodown_progan": {
        "model_path": "./weights/gandetection_resnet50nodown_progan.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
        "patch_size": None,
    },
    "gandetection_resnet50nodown_stylegan2": {
        "model_path": "./weights/gandetection_resnet50nodown_stylegan2.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
        "patch_size": None,
    },
}


def load_model(model_name, device):
    """Loads the model from the config."""
    model_config = models_config[model_name]
    model = resnet50nodown(device, model_config["model_path"])
    return model


def process_image(model, img):
    """Passes the image through the model to get the logit."""
    return model.apply(img)


def main():
    """
    The main function of the script. It parses command-line arguments and runs the inference test.

    The function expects three command-line arguments:
    - `--image_path`: The path to the image file on which inference is to be performed.
    - `--debug`: Show memory usage or not

    After parsing the arguments, it calls the `run_single_test` function to perform inference
    on the specified image using the provided model weights.
    """
    parser = argparse.ArgumentParser(
        description="This script tests the network on a single image."
    )
    parser.add_argument(
        "--image_path",
        "-i",
        type=str,
        required=True,
        help="input image path (PNG or JPEG)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="enables debug mode to print memory usage",
    )
    config = parser.parse_args()
    image_path = config.image_path
    debug_mode = config.debug

    start_time = time.time()
    logits = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processed_image_path = compress_and_resize_image(image_path)
    img = Image.open(processed_image_path).convert("RGB")
    img.load()

    for model_name in models_config:
        if debug_mode:
            print_memory_usage()

        print(f"Model {model_name} processed")

        model = load_model(model_name, device)
        logit = process_image(model, img)

        logits[model_name] = logit.item() if isinstance(logit, np.ndarray) else logit

        # Unload model from memory
        del model
        torch.cuda.empty_cache()

        if debug_mode:
            print_memory_usage()

    execution_time = time.time() - start_time

    label = "True" if any(value < 0 for value in logits.values()) else "False"

    # Construct output JSON
    output = {
        "product": "gan-model-detector",
        "detection": {
            "logit": logits,
            "IsGANImage?": label,
            "ExecutionTime": execution_time,
        },
    }

    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()
