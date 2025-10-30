import ray
import numpy as np
import os
from preprocess import preprocess_batch


def main():
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.join(base_dir, "..", "data", "raw", "hsf_page")

    # Get all image paths
    image_lst = [
        os.path.join(image_dir, f)
        for image_dir, _, files in os.walk(image_dir)
        for f in files
    ]

    batch_size = 30

    batch = preprocess_batch(image_lst[:50])

    #ray.init()


if __name__ == '__main__':
    main()