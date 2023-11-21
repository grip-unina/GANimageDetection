import os
import time
import argparse
import psutil
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
    "Grag2021_progan": {
        "model_path": "./weights/gandetection_resnet50nodown_progan.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
        "patch_size": None,
    }
}

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

    if debug_mode:
        print_memory_usage()

    from torch.cuda import is_available as is_available_cuda

    device = "cuda:0" if is_available_cuda() else "cpu"

    net = resnet50nodown(device, models_config["Grag2021_progan"]["model_path"])

    if debug_mode:
        print_memory_usage()

    print("GAN IMAGE DETECTION")
    print("START")

    tic = time.time()
    img = Image.open(image_path).convert("RGB")
    img.load()
    logit = net.apply(img)
    toc = time.time()

    print(f"Image: {image_path}, Logit: {logit}, Time: {toc-tic:.2f}s")

    print("\nDONE")
