import cv2
import numpy as np
import ray
from labels import get_label_dic


#@ray.remote
def get_characters(batch):
    for path in batch:
        image = greyscale(path)
        label_dic = get_label_dic(path)
        image, mask = find_boxes(image)
        boxes = extract_boxes(image, mask)
        cleaned_boxes = clean_boxes(boxes)
        for box in cleaned_boxes:
            cropped = crop(box['image'])
            label = find_label(box, label_dic)
            chars = segment_chars(cropped, label)

def greyscale(path): return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def light_clean(image): return cv2.medianBlur(image, 3)

def deskew(image): 
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal edges
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, horizontal_kernel)

    # Edge detection
    edges = cv2.Canny(h_lines, 50, 150, apertureSize=3)

    # Hough lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    angles = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90  # convert to degrees relative to horizontal
            if -45 < angle < 45:  # consider only near-horizontal
                angles.append(angle)

    if len(angles) == 0:
        skew_angle = 0
    else:
        skew_angle = np.median(angles)


    # Rotate image
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed


def find_boxes(image):
    MIN_HORIZ_LEN = 70
    MIN_VERT_LEN  = 70
    WINDOW_WIDTH  = 1000
    WINDOW_HEIGHT = 800
    LINE_GAP = 3
    MIN_BOX_AREA = 1000
    MAX_BOX_AREA_RATIO = 0.3
    HORIZ_DILATE = 20
    
 
    # Convert to grayscale
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
 
    # Remove small text components before detecting lines
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    text_mask = np.zeros_like(th)
    for i in range(1, num_labels):
        x, y, w_box, h_box, area = stats[i]
        if area > 200:
            text_mask[labels == i] = 255
    
 
    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MIN_HORIZ_LEN, 1))
    h_lines = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, h_kernel)
    h_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_GAP, 1))
    h_lines = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, h_close_kernel)
    h_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZ_DILATE, 1))
    h_lines = cv2.dilate(h_lines, h_dilate_kernel)
    
 
    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, MIN_VERT_LEN))
    v_lines = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, v_kernel)
    v_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_GAP))
    v_lines = cv2.morphologyEx(v_lines, cv2.MORPH_CLOSE, v_close_kernel)
    
 
    # Find form structure
    #intersections = cv2.bitwise_and(h_lines, v_lines)
    form_structure = cv2.bitwise_or(h_lines, v_lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    form_structure = cv2.dilate(form_structure, kernel, iterations=1)
    
 
    # Find contours on the connected line mask
    contours, hierarchy = cv2.findContours(form_structure, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                area = w_box * h_box
                if MIN_BOX_AREA < area < MAX_BOX_AREA_RATIO * gray.shape[0] * gray.shape[1]:
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
    """
    # Visualize boxes
    orig = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    contours_to_draw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_to_draw:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        cv2.rectangle(orig, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
 
    # Display
    cv2.namedWindow("Boxes detected", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Boxes detected", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.imshow("Boxes detected", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return gray, mask


def extract_boxes(gray_image, mask, descender_padding=15):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Add padding for descenders (j, g, p, q, y)
        y_end = min(y + h + descender_padding, gray_image.shape[0])
        box_image = gray_image[y:y_end, x:x+w].copy()
        
        boxes.append({
            'id': idx,
            'image': box_image,
            'bbox': (x, y, w, h + descender_padding),
            'position': (x, y)
        })
    
    # Sort boxes by position (top-to-bottom, left-to-right)
    boxes.sort(key=lambda b: (b['position'][1], b['position'][0]))
    return boxes

def display_boxes_one_by_one(boxes, area):
    for box in boxes:
        cv2.imshow("Box", box['image'])
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_box_area(box):
    height, width = box.shape[:2]
    area = height * width
    cv2.imshow(str(area), box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def remove_horizontal_lines(box, min_line_length):
    _, binary = cv2.threshold(box, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # Vertical dilation
    detected_lines = cv2.dilate(detected_lines, dilate_kernel, iterations=2)

    cleaned = cv2.subtract(binary, detected_lines)
    return cleaned


def clean_boxes(boxes):
    MIN_LINE_LENGTH = 80
    cleaned_boxes = []
        
    for box in boxes:
        cleaned_image = remove_horizontal_lines(
            box['image'], 
            MIN_LINE_LENGTH,
        )
        if box['position'][1] >= 650:
            cleaned_boxes.append({
            'id': box['id'],
            'image': cleaned_image,
            'original_image': box['image'],
            'bbox': box['bbox'],
            'position': box['position']
            })
        #display_box_area(cleaned_image)
    #display_boxes_one_by_one(cleaned_boxes, area)
    return cleaned_boxes


def segment_chars(image, char_labels):
    if image.ndim != 2:
        raise ValueError("Image must be 2D grayscale")
    if not char_labels:
        raise ValueError("char_labels must not be empty")

    H, W = image.shape
    N = len(char_labels)

    # If only 1 character, no segmentation needed
    if N == 1:
        boxes = [(0, W)]
    else:
        # Compute vertical projection
        proj = np.sum(image, axis=0).astype(np.float32)  # shape (W,)

        # Normalize and smooth slightly to reduce noise
        proj = cv2.GaussianBlur(proj[None, :], (1, 5), 0).flatten()

        # Find candidate split points (valleys)
        min_peak_distance = max(1, W // (2 * N)) 

        # Use simple valley detection: points lower than neighbors
        candidates = []
        for i in range(1, W - 1):
            if proj[i] < proj[i - 1] and proj[i] <= proj[i + 1]:
                # Also require it's relatively low
                if proj[i] < np.percentile(proj, 30):
                    candidates.append(i)

        # Remove candidates too close to each other
        candidates.sort()
        filtered = []
        last = -min_peak_distance
        for c in candidates:
            if c - last >= min_peak_distance:
                filtered.append(c)
                last = c
        candidates = filtered

        # If we have enough candidates, use the best (N-1) ones
        if len(candidates) >= N - 1:
            # Choose the lowest valleys (most likely real gaps)
            candidate_vals = [(proj[c], c) for c in candidates]
            candidate_vals.sort()  # lowest density first
            splits = sorted([c for _, c in candidate_vals[:N - 1]])
        else:
            # Not enough natural gaps
            splits = candidates.copy()
            # Create initial segments based on current splits
            segments = []
            prev = 0
            for s in splits:
                segments.append((prev, s))
                prev = s
            segments.append((prev, W))

            # Now iteratively split the widest segment until we have N segments
            while len(segments) < N:
                # Find widest segment
                widths = [e - s for s, e in segments]
                idx = np.argmax(widths)
                s, e = segments[idx]
                mid = (s + e) // 2
                # Replace segment with two halves
                segments[idx] = (s, mid)
                segments.insert(idx + 1, (mid, e))

            # Extract split points
            splits = [seg[1] for seg in segments[:-1]]

        # Build boxes from splits
        boxes = []
        prev = 0
        for split in splits:
            boxes.append((prev, split))
            prev = split
        boxes.append((prev, W))

    # Ensure exactly N boxes, if not do equal width boxes
    if len(boxes) != N:
        w = W / N
        boxes = [(int(i * w), int((i + 1) * w)) for i in range(N)]

    # Clip to valid range
    boxes = [(max(0, s), min(W, e)) for s, e in boxes if e > s]

    # Draw box over original image
    img_disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colour = (0, 255, 255)
    for i in range(N):
        x1, x2 = boxes[i]
        cv2.rectangle(img_disp, (x1, 0), (x2, H), colour, thickness=2)
        cv2.putText(img_disp, char_labels[i], (x1 + 2, H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    # Show
    cv2.imshow("Segmentation (Fixed to Label Length)", img_disp)
    print(f"Segmented into {N} characters: '{char_labels}'")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop(image, min_component_area: int = 100):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=4, ltype=cv2.CV_32S
    )

    # [x, y, w, h, area] for each component (label 0 = background)
    content_mask = np.zeros_like(image, dtype=np.uint8)

    found_content = False
    for i in range(1, num_labels):  
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            content_mask[labels == i] = 255
            found_content = True

    if not found_content:
        cropped = image
    else:
        # Get tight bounding box around denoised content
        coords = cv2.findNonZero(content_mask)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = image[y:y + h, x:x + w]

    return cropped
    """
    # display
    cv2.imshow("Cropped (Noise-Ignored)", cropped)
    print(f"Original: {image.shape} → Cropped: {cropped.shape}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

def find_label(box, dic):
    # Expecting dictionary with image, position, etc...
    height, width = box['image'].shape[:2]
    area = height * width
    x, y = box['position']

    closest_y = closest(dic.keys(), y)
    closest_x = closest(dic[closest_y].keys(), x)
    label = dic[closest_y][closest_x]

    return label

def closest(lst, x):
    if not lst:
        raise ValueError("Empty list")
    return min(lst, key=lambda v: abs(v - x))
