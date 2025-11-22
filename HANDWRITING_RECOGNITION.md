# Handwriting Recognition Feature

## Overview

The handwriting recognition feature allows you to test your trained federated learning model directly in the browser! After completing a training session, a new **Handwriting Recognition** node will automatically appear, connected to your trained model.

## How It Works

### 1. **Train a Model**
   - Configure your dataset, model, and training parameters
   - Run the federated learning training
   - Wait for training to complete

### 2. **Automatic Node Creation**
   - When training completes, two nodes appear:
     - **Results Node**: Shows training metrics and confusion matrix
     - **Handwriting Recognition Node**: A drawing canvas for testing your model

### 3. **Draw and Predict**
   - Use your mouse to draw a digit (0-9) on the canvas
   - Click the **"🔮 Predict"** button
   - The model will predict which digit you drew
   - You'll see:
     - The predicted digit (large display)
     - Top 3 probabilities with confidence scores

### 4. **Clear and Retry**
   - Click **"Clear"** to erase the canvas
   - Draw a new digit and predict again

## Features

### Drawing Canvas
- **Size**: 280×280 pixels
- **Brush**: Smooth, rounded brush with 10px radius
- **Background**: White (inverted during preprocessing)

### Image Preprocessing
The canvas image is preprocessed using the same technique as `playground.py`:
1. **Grayscale conversion** and **inversion**
2. **Gaussian blur** (radius 0.5)
3. **Max filter** for dilation
4. **Binarization** (threshold 128)
5. **Cropping** to bounding box
6. **Resizing** to fit 20×20 box
7. **Centering** in 28×28 canvas
8. **Center of mass adjustment**
9. **Deskewing** using second-order moments

This ensures your handwritten digits are processed identically to MNIST training data.

## API Endpoint

The feature uses a new API endpoint:

```
POST /api/predict/digit
```

**Form Data:**
- `image`: The canvas image as a PNG file
- `session_id`: The training session ID

**Response:**
```json
{
  "prediction": 7,
  "probabilities": [0.001, 0.002, ..., 0.95, ...]
}
```

## Browser vs Desktop

| Feature | Browser (`HandwritingNode`) | Desktop (`playground.py`) |
|---------|----------------------------|---------------------------|
| Interface | HTML5 Canvas | Tkinter Canvas |
| Preprocessing | Backend (Python/PIL) | Local (Python/PIL) |
| Model Loading | Automatic by session | Manual file selection |
| Drawing | Mouse | Mouse |
| Predictions | Real-time via API | Local prediction |

## Technical Details

### Frontend (`HandwritingNode.tsx`)
- React component with HTML5 Canvas
- Mouse event handlers for drawing
- Converts canvas to PNG blob
- Sends to API via FormData
- Displays prediction results

### Backend (`api_server.py`)
- New `/api/predict/digit` endpoint
- Loads trained model from session outputs
- Applies identical preprocessing as `playground.py`
- Returns prediction and probabilities

## Example Workflow

1. **Start training** with 3 clients on MNIST dataset
2. **Wait for completion** (~2-3 minutes)
3. **See Results Node** with accuracy/loss metrics
4. **See Handwriting Node** appear below Results
5. **Draw digit "5"** on canvas
6. **Click Predict** → Model predicts: **5** (98.7% confidence)
7. **Clear and draw "3"** → Model predicts: **3** (95.2% confidence)

## Notes

- The handwriting node uses the **exact model** from your training session
- Each training session gets its own model (stored in `outputs/[session_id]/model.keras`)
- Preprocessing is **identical** to the desktop `playground.py` for consistency
- The canvas is **280×280** pixels for comfortable drawing, then preprocessed to **28×28** for the model

## Troubleshooting

### "Model not found for this session"
- Ensure training completed successfully
- Check that `outputs/[session_id]/model.keras` exists

### "Failed to predict digit"
- Verify API server is running (`python api_server.py`)
- Check console for errors
- Ensure PIL (Pillow) is installed: `pip install pillow`

### Poor predictions
- Try drawing larger, centered digits
- Use bold strokes (brush is 10px radius)
- Avoid very thin or small digits
- Remember: model is trained on MNIST-style digits

## Future Enhancements

Potential improvements:
- Touch support for tablets/mobile
- Adjustable brush size
- Save/load drawings
- Batch prediction mode
- Confidence threshold visualization
- Real-time prediction (as you draw)

