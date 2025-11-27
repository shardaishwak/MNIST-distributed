import ray
import os
from segment import get_characters


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
 
    batch = get_characters(image_lst[:15])


    #ray.init()


if __name__ == '__main__':
    main()