import os
import time
import argparse
import json
import psutil
import torch
import numpy as np

from PIL import Image
from resnet50nodown import resnet50nodown


def print_memory_usage():
    """
    Prints the current memory usage of the process.
    """
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
    model_config = models_config[model_name]
    model = resnet50nodown(device, model_config["model_path"])
    return model


def process_image(model, img):
    return model.apply(img)


if __name__ == "__main__":
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

    from torch.cuda import is_available as is_available_cuda

    device = "cuda:0" if is_available_cuda() else "cpu"
    img = Image.open(image_path).convert("RGB")
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
