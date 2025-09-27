
import sys
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import numpy as np
import tensorflow as tf
import math

CANVAS_SIZE = 280         
BRUSH_RADIUS = 10         
MODEL_DEFAULT = "aggregated_mnist_model.keras"  

def load_model(path: str | None):
    
    if path is None:
        if os.path.exists("aggregated_mnist_model.keras"):
            path = "aggregated_mnist_model.keras"
        elif os.path.exists("aggregated_mnist_model.h5"):
            path = "aggregated_mnist_model.h5"
        else:
            raise FileNotFoundError("No aggregated model found (.keras or .h5) in current directory.")
    model = tf.keras.models.load_model(path)
    
    return model

DESKEW = True  

def _center_of_mass(img_arr):
    
    y_idx, x_idx = np.indices(img_arr.shape)
    m = img_arr.sum() + 1e-8
    cy = (y_idx * img_arr).sum() / m
    cx = (x_idx * img_arr).sum() / m
    return cy, cx

def _shift_to_center(img_arr, target=(14, 14)):
    cy, cx = _center_of_mass(img_arr)
    dy = int(round(target[0] - cy))
    dx = int(round(target[1] - cx))
    shifted = np.roll(img_arr, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)
    return shifted

def _deskew_28(img_arr):
    """
    Simple deskew by second-order moments (like common MNIST preproc).
    Keeps it light and PIL-free to avoid dependencies.
    """
    
    y, x = np.indices(img_arr.shape)
    img = img_arr
    img /= (img.max() + 1e-8)

    m00 = img.sum() + 1e-8
    m10 = (x * img).sum()
    m01 = (y * img).sum()
    x_cent = m10 / m00
    y_cent = m01 / m00

    x = x - x_cent
    y = y - y_cent

    mu11 = (x * y * img).sum() / m00
    mu20 = ((x ** 2) * img).sum() / m00
    mu02 = ((y ** 2) * img).sum() / m00
    
    denom = (mu20 - mu02)
    if abs(denom) < 1e-8:
        return img_arr  

    tan2theta = 2 * mu11 / denom
    theta = 0.5 * math.atan(tan2theta)
    t = math.tan(theta)
    h, w = img_arr.shape
    yy, xx = np.indices((h, w)).astype(np.float32)
    yy -= h / 2.0
    xx -= w / 2.0
    yy_prime = yy - t * xx
    yy_prime += h / 2.0
    xx += w / 2.0

    
    def bilinear(img, yyf, xxf):
        h, w = img.shape
        y0 = np.clip(np.floor(yyf).astype(int), 0, h - 1)
        x0 = np.clip(np.floor(xxf).astype(int), 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)

        wy = yyf - y0
        wx = xxf - x0

        top = (1 - wx) * img[y0, x0] + wx * img[y0, x1]
        bot = (1 - wx) * img[y1, x0] + wx * img[y1, x1]
        return (1 - wy) * top + wy * bot

    deskewed = bilinear(img_arr, yy_prime, xx)
    return deskewed

def preprocess_pil_digit(pil_img: Image.Image) -> np.ndarray:
    """
    Robust MNIST-like preprocessing with dilation + centering + (optional) deskew.
    Saves the final 28x28 input as 'last_processed.png' for inspection.
    """
    img = pil_img.convert("L")
    img = ImageOps.invert(img)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img = img.filter(ImageFilter.MaxFilter(size=3))

    arr = np.array(img, dtype=np.uint8)
    th = 128
    bin_mask = (arr >= th)

    if not bin_mask.any():
        x28 = np.zeros((28, 28, 1), dtype=np.float32)
        Image.fromarray((x28[:, :, 0] * 255).astype(np.uint8)).save("last_processed.png")
        return x28[np.newaxis, ...]

    ys, xs = np.where(bin_mask)
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    img = img.crop((minx, miny, maxx + 1, maxy + 1))
    w, h = img.size
    scale = 20 / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(img, (left, top))

    
    x = np.array(canvas).astype("float32") / 255.0
    x = _shift_to_center(x)
    if DESKEW:
        x = _deskew_28(x)

    out_vis = (x * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(out_vis).save("last_processed.png")

    x = x.reshape(1, 28, 28, 1).astype(np.float32)
    return x

class DigitPlayground(tk.Tk):
    def __init__(self, model_path: str | None = None):
        super().__init__()
        self.title("MNIST Playground")
        self.resizable(False, False)
        self.model = load_model(model_path)
        self._build_ui()
        self.pil = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.pil)

        self.last_xy = None

    def _build_ui(self):
        frame = ttk.Frame(self, padding=8)
        frame.grid(row=0, column=0)

        self.canvas = tk.Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", cursor="cross")
        self.canvas.grid(row=0, column=0, rowspan=6, padx=(0, 8))

        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

        ttk.Button(frame, text="Predict", command=self.predict).grid(row=0, column=1, sticky="ew")
        ttk.Button(frame, text="Clear", command=self.clear).grid(row=1, column=1, sticky="ew")
        ttk.Button(frame, text="Save Drawing", command=self.save_image).grid(row=2, column=1, sticky="ew")

        ttk.Separator(frame, orient="horizontal").grid(row=3, column=1, sticky="ew", pady=6)

        self.pred_label = ttk.Label(frame, text="Draw a digit, then click Predict.", font=("Helvetica", 12))
        self.pred_label.grid(row=4, column=1, sticky="ew")

        self.prob_label = ttk.Label(frame, text="", font=("Courier", 10))
        self.prob_label.grid(row=5, column=1, sticky="ew")

    def on_down(self, event):
        self.last_xy = (event.x, event.y)
        self._draw_point(event.x, event.y)

    def on_move(self, event):
        if self.last_xy is None:
            return
        x0, y0 = self.last_xy
        x1, y1 = event.x, event.y
        
        self.canvas.create_line(x0, y0, x1, y1, width=BRUSH_RADIUS * 2, capstyle=tk.ROUND, smooth=True)
        self.draw.line((x0, y0, x1, y1), fill="black", width=BRUSH_RADIUS * 2, joint="curve")
        
        r = BRUSH_RADIUS
        self.canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, fill="black", outline="black")
        self.draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill="black", outline="black")
        self.last_xy = (x1, y1)

    def on_up(self, _event):
        self.last_xy = None

    def _draw_point(self, x, y):
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill="black", outline="black")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill="white")
        self.pred_label.config(text="Cleared. Draw a digit, then click Predict.")
        self.prob_label.config(text="")

    def save_image(self):
        self.pil.save("drawing.png")
        self.pred_label.config(text="Saved as drawing.png")

    def predict(self):
        x = preprocess_pil_digit(self.pil)
        probs = self.model.predict(x, verbose=0)[0]  
        pred = int(np.argmax(probs))
        
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_text = "\n".join([f"{i}: {probs[i]:.4f}" for i in top3_idx])
        self.pred_label.config(text=f"Prediction: {pred}")
        self.prob_label.config(text=f"Top-3 probs:\n{top3_text}")

def main():
    model_path = sys.argv[1] if len(sys.argv) >= 2 else None
    app = DigitPlayground(model_path)
    app.mainloop()

if __name__ == "__main__":
    main()
