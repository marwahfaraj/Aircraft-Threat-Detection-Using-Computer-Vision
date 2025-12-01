"""
Real-Time Aircraft Threat Detection App
Uses Gradio for webcam-based aircraft detection with threat classification
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODEL_PATH,
    PROJECT_ROOT,
    CLASSES_FILE,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    GRADIO_SHARE
)
from utils import is_threat, draw_detection, draw_status_banner

# Load model
print("Loading YOLO model...")

# Try to find model - check multiple possible locations
possible_paths = [
    MODEL_PATH,  # Primary location (YOLOv8s augmented - best model)
    PROJECT_ROOT / "results/model_1_output/augmented_on_fly/weights/best.pt",  # YOLOv8n augmented (fallback)
    PROJECT_ROOT / "results/model_2_output/yolov8s_baseline2/weights/best.pt",  # YOLOv8s baseline
    PROJECT_ROOT / "results/model_1_output/baseline_no_aug2/weights/best.pt",  # YOLOv8n baseline
    PROJECT_ROOT / "results/models_output/augmented_on_fly/weights/best.pt",  # Old location
    PROJECT_ROOT / "notebooks/yolov8n.pt",  # Pre-trained fallback
]

model_path = None
for path in possible_paths:
    if path.exists():
        model_path = path
        break

if model_path is None:
    print(f"Warning: Model not found at {MODEL_PATH}")
    print("Attempting to use pre-trained YOLOv8n model...")
    model_path = "yolov8n.pt"  # Use pre-trained model as fallback
    print("Note: This will use a generic YOLOv8n model, not your trained model!")
    
model = YOLO(str(model_path))
print(f"Model loaded from: {model_path}")

# Load class names
class_names = []
if CLASSES_FILE.exists():
    with open(CLASSES_FILE, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(class_names)} class names")
else:
    print(f"Warning: {CLASSES_FILE} not found. Using default class names.")

def process_frame(frame):
    """
    Process a single frame for aircraft detection
    
    Args:
        frame: Input image (numpy array - can be BGR from OpenCV or RGB from Gradio)
    
    Returns:
        Annotated image with bounding boxes and threat labels (RGB format for Gradio)
    """
    if frame is None:
        return None
    
    # Convert to numpy array if needed
    if hasattr(frame, 'shape'):
        img = frame.copy()
    else:
        img = np.array(frame)
    
    # Determine if frame is BGR (from OpenCV) or RGB (from Gradio)
    # OpenCV VideoCapture returns BGR, so we assume BGR if coming from video
    # But we'll check - if it's already RGB, we need to convert to BGR for OpenCV operations
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Assume BGR from OpenCV VideoCapture - keep as is for OpenCV operations
        img_bgr = img.copy()
    else:
        img_bgr = img
    
    # Run inference
    results = model.predict(
        img_bgr,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )
    
    # Get the first result (single image)
    result = results[0]
    
    # Draw bounding boxes and labels
    annotated_img = img_bgr.copy()
    threat_detected = False
    num_detections = 0
    
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get class name
            if class_id < len(class_names):
                aircraft_type = class_names[class_id]
            else:
                aircraft_type = f"Class_{class_id}"
            
            # Determine if threat
            threat = is_threat(aircraft_type)
            if threat:
                threat_detected = True
            
            # Draw detection using utility function
            annotated_img = draw_detection(
                annotated_img, x1, y1, x2, y2, 
                aircraft_type, confidence, threat
            )
            
            num_detections += 1
        
        # Draw status banner
        annotated_img = draw_status_banner(
            annotated_img, threat_detected, num_detections
        )
    
    # Convert BGR back to RGB for Gradio display
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return annotated_img_rgb

# Global variables for video streaming
is_streaming = False
video_capture = None

def get_video_frame():
    """Get and process the latest video frame"""
    global video_capture, is_streaming
    
    if not is_streaming or video_capture is None or not video_capture.isOpened():
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Click 'Start Video' to begin", 
                   (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
    
    ret, frame = video_capture.read()
    if not ret:
        return None
    
    try:
        # Process frame for detection (frame is BGR from OpenCV)
        processed = process_frame(frame)
        return processed
    except Exception as e:
        print(f"Error processing frame: {e}")
        # Return original frame converted to RGB if processing fails
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# Create Gradio interface
with gr.Blocks(title="Aircraft Threat Detection") as app:
    gr.Markdown(
        """
        # 🛩️ Aircraft Threat Detection System
        
        **Real-time video detection** using YOLOv8s. Point your webcam at aircraft (or toy aircraft) 
        to detect and classify them as threats with live bounding boxes.
        
        **Features:**
        - 🎥 Real-time continuous video streaming
        - 📦 Bounding box visualization
        - ⚠️ Threat classification
        - 🏷️ Aircraft type identification
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎥 Live Video Detection")
            video_display = gr.Image(label="Live Detection Feed", type="numpy")
            
            with gr.Row():
                start_btn = gr.Button("▶️ Start Video", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop Video", variant="stop", size="lg")
            
            status = gr.Textbox(label="Status", value="Ready. Click 'Start Video' to begin.", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### 📷 Or Use Static Image")
            image_input = gr.Image(label="Upload Image", type="numpy")
            detect_btn = gr.Button("🔍 Detect from Image", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 📊 Detection Info")
            gr.Markdown(
                """
                **How to Use:**
                1. Click "▶️ Start Video" to begin continuous live detection
                2. Point your webcam at a toy aircraft
                3. Video will stream automatically - no refresh needed!
                4. Click "⏹️ Stop Video" when done
                
                **Detection Legend:**
                - **Red boxes**: Threat detected (military aircraft)
                - **Yellow boxes**: Safe aircraft (commercial)
                - **Confidence scores** shown for each detection
                
                **Model Performance:**
                - Trained on 49,482 aircraft images
                - 195 different aircraft types
                - Best model: 67.1% mAP@0.5
                """
            )
    
    # Process uploaded image
    def process_image(img):
        if img is None:
            return None
        try:
            return process_frame(img)
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    # Start video streaming
    def start_stream():
        """Initialize video capture and start streaming"""
        global video_capture, is_streaming
        
        if is_streaming:
            return "Video already running!", None
        
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            return "❌ Could not open webcam. Please check your camera permissions.", None
        
        is_streaming = True
        print("▶️ Starting continuous video stream...")
        
        # Return first frame
        ret, frame = video_capture.read()
        if ret:
            first_frame = process_frame(frame)
        else:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Starting camera...", 
                       (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            first_frame = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        
        return "✅ Video streaming started! Point your webcam at an aircraft.", first_frame
    
    # Stop video streaming
    def stop_stream():
        """Stop video capture and streaming"""
        global video_capture, is_streaming
        
        is_streaming = False
        
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        
        print("⏹️ Video stream stopped")
        
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Video stopped", 
                   (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return "⏹️ Video detection stopped.", cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
    
    # Continuous frame update
    def stream_video():
        """Continuously stream video frames"""
        while is_streaming:
            frame = get_video_frame()
            yield frame
            time.sleep(0.033)  # ~30 FPS
    
    # Connect components
    start_btn.click(
        fn=start_stream,
        outputs=[status, video_display]
    ).then(
        fn=stream_video,
        outputs=video_display
    )
    
    stop_btn.click(
        fn=stop_stream,
        outputs=[status, video_display]
    )
    
    detect_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=video_display
    )
    
    gr.Markdown(
        """
        ---
        ### Model Information:
        - **Model**: YOLOv8s (Small) - Augmented Training
        - **Classes**: 195 aircraft types (commercial + military)
        - **Confidence Threshold**: 0.25
        - **Performance**: 67.1% mAP@0.5 on test set
        - **Speed**: ~30 FPS real-time processing
        """
    )

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting Aircraft Threat Detection App")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Classes: {len(class_names)}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("="*70 + "\n")
    
    # Try to launch on the configured port, if busy try next port
    port = GRADIO_SERVER_PORT
    max_attempts = 5
    
    for attempt in range(max_attempts):
        try:
            app.launch(
                server_name=GRADIO_SERVER_NAME,
                server_port=port,
                share=GRADIO_SHARE,
                show_error=True,
                inbrowser=(attempt == 0)  # Only open browser on first attempt
            )
            break
        except OSError as e:
            if "Cannot find empty port" in str(e) and attempt < max_attempts - 1:
                print(f"⚠️ Port {port} is busy, trying port {port + 1}...")
                port += 1
            else:
                raise
