import ray
import cv2


def main():

    ray.init()

    # Perform preprocessing: greyscale, binarize, invert, clean noise on worker nodes.
    @ray.remote
    def preprocess_batch(batch):
        pass


    




if __name__ == '__main__':
    main()