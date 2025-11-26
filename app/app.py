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
import threading
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
latest_processed_frame = None
processing_thread = None

def video_processing_loop():
    """Background thread that continuously captures and processes video frames"""
    global is_streaming, video_capture, latest_processed_frame
    
    print("🎥 Video processing thread started")
    frame_count = 0
    
    while is_streaming:
        if video_capture is None or not video_capture.isOpened():
            time.sleep(0.1)
            continue
        
        ret, frame = video_capture.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        try:
            # Process frame for detection (frame is BGR from OpenCV)
            processed = process_frame(frame)
            latest_processed_frame = processed
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"✅ Processed {frame_count} frames...")
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Store original frame converted to RGB if processing fails
            latest_processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        time.sleep(0.033)  # ~30 FPS processing
    
    print(f"🎥 Video processing thread stopped (processed {frame_count} frames)")

def get_video_frame():
    """Get the latest processed video frame"""
    global latest_processed_frame, is_streaming
    
    # If we have a processed frame, return it
    if latest_processed_frame is not None:
        return latest_processed_frame
    
    # Return placeholder based on state
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    if is_streaming:
        cv2.putText(placeholder, "Processing video...", 
                   (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(placeholder, "Click 'Start Video' to begin", 
                   (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)

def start_video():
    """Start video capture and processing"""
    global is_streaming, video_capture, processing_thread
    
    if is_streaming:
        return "Video already running!"
    
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return "❌ Could not open webcam. Please check your camera permissions."
    
    is_streaming = True
    
    # Start background processing thread
    processing_thread = threading.Thread(target=video_processing_loop, daemon=True)
    processing_thread.start()
    
    print("▶️ Starting video detection...")
    return "✅ Video detection started! Point your webcam at an aircraft."

def stop_video():
    """Stop video capture"""
    global is_streaming, video_capture, latest_processed_frame, processing_thread
    
    is_streaming = False
    
    # Wait for thread to finish
    if processing_thread is not None:
        processing_thread.join(timeout=1.0)
        processing_thread = None
    
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    
    latest_processed_frame = None
    print("⏹️ Stopping video detection...")
    
    return "⏹️ Video detection stopped."

# Create Gradio interface
with gr.Blocks(title="Aircraft Threat Detection") as app:
    gr.Markdown(
        """
        # 🛩️ Aircraft Threat Detection System
        
        **Real-time video detection** using YOLOv8s. Point your webcam at aircraft (or toy aircraft) 
        to detect and classify them as threats with live bounding boxes.
        
        **Features:**
        - 🎥 Real-time video streaming
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
            update_btn = gr.Button("🔄 Refresh Frame", variant="secondary")
            
            gr.Markdown("---")
            gr.Markdown("### 📷 Or Use Static Image")
            image_input = gr.Image(label="Upload Image", type="numpy")
            detect_btn = gr.Button("🔍 Detect from Image", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 📊 Detection Info")
            gr.Markdown(
                """
                **How to Use:**
                1. Click "▶️ Start Video" to begin live detection
                2. Point your webcam at a toy aircraft
                3. Click "🔄 Refresh Frame" repeatedly to see real-time updates
                4. Or hold down the refresh button for continuous updates
                5. Click "⏹️ Stop Video" when done
                
                **Tip:** For smoother real-time viewing, click the refresh button rapidly or use a browser extension to auto-click it.
                
                **Detection Legend:**
                - **Red boxes**: Threat detected
                - **Yellow boxes**: Aircraft detected (non-threat)
                - **Confidence scores** shown for each detection
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
    
    # Connect components
    def start_and_update():
        """Start video and return first frame"""
        status_msg = start_video()
        # Wait a moment for first frame to be processed
        time.sleep(0.3)
        frame = get_video_frame()
        if frame is None:
            # Return placeholder if no frame yet
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Starting camera...", 
                       (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            frame = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        return status_msg, frame
    
    start_btn.click(
        fn=start_and_update,
        outputs=[status, video_display]
    )
    
    def stop_and_clear():
        """Stop video and clear display"""
        status_msg = stop_video()
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Video stopped", 
                   (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return status_msg, cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
    
    stop_btn.click(
        fn=stop_and_clear,
        outputs=[status, video_display]
    )
    
    detect_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=video_display
    )
    
    # Simple refresh function
    def refresh_frame():
        """Refresh the video display with latest frame"""
        frame = get_video_frame()
        return frame
    
    update_btn.click(
        fn=refresh_frame,
        outputs=video_display
    )
    
    # Remove complex JavaScript - use simpler approach
    # User can click refresh button manually or we'll add a simpler auto-refresh
    
    gr.Markdown(
        """
        ---
        ### Model Information:
        - Model: YOLOv8s (Small) - Augmented Training
        - Classes: 195 aircraft types
        - Confidence Threshold: 0.25
        - **Better accuracy than YOLOv8n!**
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
