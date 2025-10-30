import cv2
import numpy as np
from skimage.transform import rotate
import ray

"""
Preprocess Order:
1. Greyscale
2. Light noise cleaning
3. Binarize (skip). Dataset already binarized
4. Detect skew
5. Rotate image
6. Remove form lines
"""
#@ray.remote
def preprocess_batch(batch):
    ret = []
    for path in batch:
        image = greyscale(path)
        #image = light_clean(image)
        #skew = find_skew(image)
        #image = rotate_image(image, skew)
        remove_form_lines(image)
        ret.append(image)



def greyscale(path): return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def light_clean(image): return cv2.medianBlur(image, 3)

def find_skew(image, delta=1, limit=5): 
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    scores = []
    angles = np.arange(-limit, limit+delta, delta)
    for angle in angles:
        rotated = rotate(thresh, angle, resize=True, order=0)
        hist = np.sum(rotated, axis=1)
        score = np.sum((hist[1:] - hist[:-1])**2)
        scores.append(score)

    best_angle = angles[np.argmax(scores)]
    return best_angle
    
    

def rotate_image(image, skew):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, skew, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    cv2.imshow("Original Image", image)
    cv2.imshow("Rotated", rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


def remove_form_lines(image):
    _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    v_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel)

    boxes = cv2.bitwise_or(h_lines, v_lines)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    boxes_dilated = cv2.dilate(boxes, kernel, iterations=1)

    contours, _ = cv2.findContours(boxes_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(th)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

    clean = cv2.bitwise_and(th, mask)
    clean = cv2.bitwise_and(clean, cv2.bitwise_not(boxes_dilated))
    
    scale_percent = 50
    width = int(clean.shape[1] * scale_percent / 100)
    height = int(clean.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(clean, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("Clean Image", resized)

    cv2.waitKey(0)  
    cv2.destroyAllWindows()  