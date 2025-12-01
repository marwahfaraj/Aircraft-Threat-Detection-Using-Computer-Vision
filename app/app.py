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


# Create Gradio interface with custom theme
custom_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#f8fafc",
    body_background_fill_dark="#0f172a",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="white",
    block_title_text_color="#1e293b",
    block_label_text_color="#475569",
    input_background_fill="white",
)

# Create custom CSS for modern look
custom_css = """
#header {
    text-align: center;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}
#header h1 {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
}
#header p {
    color: #e0f2fe;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}
.feature-box {
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
#logo-img {
    max-width: 400px;
    margin: 1rem auto;
    display: block;
    border-radius: 10px;
}
"""

with gr.Blocks(title="Aircraft Threat Detection", theme=custom_theme, css=custom_css) as app:
    # Header with logo
    with gr.Row(elem_id="header"):
        gr.Markdown(
            """
            # ✈️ Aircraft Threat Detection System
            **Real-Time Detection Using YOLOv8s Deep Learning**
            
            Advanced computer vision system for identifying and classifying aircraft threats
            """
        )
    
    # Add logo image if exists
    logo_path = PROJECT_ROOT / "aircraft_threat_detect.png"
    if logo_path.exists():
        gr.Image(str(logo_path), label=None, show_label=False, elem_id="logo-img", container=False)
    
    with gr.Row():
        # Left Column - Video Detection
        with gr.Column(scale=2):
            gr.Markdown("### 🎥 Live Video Detection", elem_classes="feature-box")
            video_display = gr.Image(
                label="Live Detection Feed", 
                type="numpy",
                height=480,
                show_label=True
            )
            
            with gr.Row():
                start_btn = gr.Button(
                    "▶️ Start Video Detection", 
                    variant="primary", 
                    size="lg",
                    scale=2
                )
                stop_btn = gr.Button(
                    "⏹️ Stop", 
                    variant="stop", 
                    size="lg",
                    scale=1
                )
            
            status = gr.Textbox(
                label="System Status", 
                value="🟢 Ready - Click 'Start Video Detection' to begin",
                interactive=False,
                show_label=True
            )
            
            gr.Markdown("---")
            gr.Markdown("### 📷 Static Image Upload", elem_classes="feature-box")
            image_input = gr.Image(
                label="Upload Aircraft Image", 
                type="numpy",
                height=300
            )
            detect_btn = gr.Button(
                "🔍 Analyze Image", 
                variant="primary",
                size="lg"
            )
        
        # Right Column - Information Panel
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Instructions", elem_classes="feature-box")
            gr.Markdown(
                """
                **Live Video Mode:**
                1. Click **Start Video Detection**
                2. Point webcam at aircraft
                3. View real-time detections
                4. Click **Stop** when finished
                
                **Image Upload Mode:**
                1. Upload aircraft image
                2. Click **Analyze Image**
                3. View detection results
                """
            )
            
            gr.Markdown("### 🎯 Detection Legend", elem_classes="feature-box")
            gr.Markdown(
                """
                🔴 **Red Box** = Threat Detected  
                (Military aircraft)
                
                🟡 **Yellow Box** = Safe Aircraft  
                (Commercial aircraft)
                
                Each detection shows:
                - Aircraft type
                - Confidence score
                - Threat status
                """
            )
            
            gr.Markdown("### 📊 Model Details", elem_classes="feature-box")
            gr.Markdown(
                """
                **Architecture:** YOLOv8s  
                **Training Data:** 49,482 images  
                **Aircraft Types:** 195 classes  
                **Accuracy:** 67.1% mAP@0.5  
                **Speed:** ~30 FPS real-time
                
                **Categories:**
                - Commercial aircraft
                - Military fighters
                - Bombers & transport
                - Helicopters & UAVs
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
    
    # Footer
    gr.Markdown(
        """
        ---
        <div style='text-align: center; color: #64748b; padding: 1rem;'>
        <p><strong>Aircraft Threat Detection System</strong> | CS 521 - Computer Vision | University of San Diego</p>
        <p>Powered by YOLOv8s Deep Learning Model | Real-Time Object Detection</p>
        </div>
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
