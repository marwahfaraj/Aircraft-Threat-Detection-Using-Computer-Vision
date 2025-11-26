# Aircraft Threat Detection - Deployment Guide

## Real-Time Webcam Detection App

This app provides real-time aircraft threat detection using your trained YOLOv8 model.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install Gradio separately:
```bash
pip install gradio>=4.0.0
```

### 2. Run the App

From the project root:
```bash
cd app
python app.py
```

Or from project root:
```bash
python app/app.py
```

The app will start on `http://localhost:7860`

### 3. Use the App

- **Webcam Mode**: Click on the webcam feed to start real-time detection
- **Image Mode**: Upload an image and click "Detect Aircraft"
- Point your webcam at a toy aircraft to test!

## Features

- ✅ Real-time webcam detection
- ✅ Bounding box visualization
- ✅ Threat classification
- ✅ Aircraft type identification
- ✅ Confidence scores
- ✅ Clean, user-friendly interface

## Configuration

Edit `app.py` to customize:

1. **Model Path**: Change `MODEL_PATH` to use a different model
   ```python
   MODEL_PATH = Path("results/models_output/augmented_on_fly/weights/best.pt")
   ```

2. **Confidence Threshold**: Adjust detection sensitivity
   ```python
   CONFIDENCE_THRESHOLD = 0.25  # Lower = more detections, Higher = fewer but more confident
   ```

3. **Threat Classification**: Customize which aircraft are threats
   ```python
   def is_threat(aircraft_class):
       # Option 1: All aircraft are threats
       return True
       
       # Option 2: Only military aircraft
       # military_keywords = ['F-', 'B-', 'A-10', 'AH', 'CH', 'UH']
       # return any(keyword in aircraft_class for keyword in military_keywords)
   ```

## Deployment Options

### Local Deployment
```bash
python app.py
```

### Public Link (Temporary)
```python
app.launch(share=True)  # Creates a public Gradio link
```

### Hugging Face Spaces
1. Create a Hugging Face account
2. Create a new Space
3. Upload `app.py` and model files
4. Add `requirements.txt`
5. Deploy!

### Custom Server
```python
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)
```

## Troubleshooting

### Webcam Not Working
- Check camera permissions
- Try a different browser
- Use image upload mode instead

### Model Not Found
- Ensure model path is correct
- Check that `best.pt` exists in the specified location

### Slow Performance
- Reduce image resolution
- Increase confidence threshold
- Use a smaller model (YOLOv8n instead of YOLOv8s/m)

## Model Performance

Current model (YOLOv8n Augmented):
- mAP@0.5: 53.1%
- Precision: 47.1%
- Recall: 53.2%

For better accuracy, wait for YOLOv8s or YOLOv8m training to complete.

## Next Steps

1. Complete YOLOv8s and YOLOv8m training
2. Compare models and use the best one
3. Fine-tune confidence threshold for your use case
4. Customize threat classification logic
5. Deploy to production server

