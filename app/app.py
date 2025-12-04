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
    # Header with logo
    gr.HTML("""
        <div style='text-align: center; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                    padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; font-size: 2.5rem; margin: 0;'>✈️ Aircraft Threat Detection System</h1>
            <p style='color: #e0f2fe; font-size: 1.1rem; margin-top: 0.5rem;'>
                Real-Time Detection Using YOLOv8s Deep Learning
            </p>
            <p style='color: #bfdbfe; font-size: 0.95rem;'>
                Advanced computer vision system for identifying and classifying aircraft threats
            </p>
        </div>
    """)
    
    # Add logo image if exists (centered and smaller, no container)
    logo_path = Path(__file__).parent / "aircraft_threat_detection_system.png"
    if logo_path.exists():
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                gr.Image(str(logo_path), label=None, show_label=False, height=150, container=False)
            with gr.Column(scale=1):
                pass
    
    with gr.Row():
        # Left Column - Video Detection
        with gr.Column(scale=2):
            gr.HTML("<h3 style='color: #1e3a8a;'>🎥 Live Video Detection</h3>")
            video_display = gr.Image(
                label="Live Detection Feed", 
                type="numpy",
                height=480,
                show_label=True
            )
            
            gr.HTML("""
                <style>
                    .blue-button { 
                        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important; 
                        color: white !important;
                        border: none !important;
                        padding: 0.75rem 1.5rem !important;
                        border-radius: 6px !important;
                        font-size: 1.1rem !important;
                        font-weight: 600 !important;
                        cursor: pointer !important;
                        width: 100% !important;
                        margin: 0.25rem 0 !important;
                    }
                    .blue-button:hover {
                        opacity: 0.9 !important;
                    }
                </style>
            """)
            with gr.Row():
                start_btn = gr.Button(
                    "▶️ Start Video Detection", 
                    elem_classes="blue-button",
                    size="lg",
                    scale=2
                )
                stop_btn = gr.Button(
                    "⏹️ Stop", 
                    elem_classes="blue-button",
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
            gr.HTML("<h3 style='color: #1e3a8a;'>📷 Static Image Upload</h3>")
            image_input = gr.Image(
                label="Upload Aircraft Image", 
                type="numpy",
                height=300
            )
            detect_btn = gr.Button(
                "🔍 Analyze Image", 
                elem_classes="blue-button",
                size="lg"
            )
            
            gr.Markdown("---")
            gr.HTML("<h3 style='color: #1e3a8a;'>🎬 Video Upload & Detection</h3>")
            gr.Markdown("""
                **Instructions:**
                1. Upload a video file
                2. Click 'Process Video' button
                3. Wait for processing (may take time for long videos)
                4. Processed video will appear below with detections
            """)
            video_input = gr.Video(label="Upload Video File")
            process_video_btn = gr.Button(
                "🎥 Process Video", 
                elem_classes="blue-button",
                size="lg"
            )
            video_status = gr.Textbox(
                label="Processing Status", 
                value="Upload a video and click 'Process Video'",
                interactive=False
            )
            video_output = gr.Video(label="Processed Video with Detections", width=640, height=480)
        
        # Right Column - Information Panel
        with gr.Column(scale=1):
            gr.HTML("<h3 style='color: #1e3a8a;'>📋 Instructions</h3>")
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
            
            gr.HTML("<h3 style='color: #1e3a8a;'>🎯 Detection Legend</h3>")
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
            
            gr.HTML("<h3 style='color: #1e3a8a;'>📊 Model Details</h3>")
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
    
    # Process uploaded video with temporal smoothing
    def process_video(video_path):
        """Process video file frame-by-frame with aircraft detection and smoothing"""
        if video_path is None:
            return "❌ Please upload a video first", None
        
        try:
            import tempfile
            
            status_msg = f"🎬 Starting video processing..."
            print(f"📹 Processing video: {video_path}")
            
            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("❌ Could not open video file")
                return "❌ Could not open video file. Try a different format.", None
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
            status_msg = f"📊 Video: {width}x{height}, {fps} FPS, {total_frames} frames\n⏳ Processing..."
            
            # Create temporary output video file with browser-compatible codec
            output_path = tempfile.mktemp(suffix='.mp4')
            
            # Try H.264 codec (most browser-compatible)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Fallback to mp4v if avc1 doesn't work
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                return "❌ Could not create output video", None
            
            frame_count = 0
            detections_found = 0
            
            # Temporal smoothing - keep track of recent detections
            detection_history = []
            history_size = 5  # Keep last 5 frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with detection
                processed_frame = process_frame(frame)
                
                # Convert RGB back to BGR for video writer
                processed_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                out.write(processed_bgr)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            # Release resources
            cap.release()
            out.release()
            
            status_msg = f"✅ Processing complete!\n📹 Processed {frame_count} frames\n🎯 Video ready with detections"
            print(f"✅ Video processing complete! Saved to: {output_path}")
            
            return status_msg, output_path
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"❌ Error processing video: {e}")
            import traceback
            traceback.print_exc()
            return error_msg, None
    
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
    
    process_video_btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=[video_status, video_output]
    )
    
    # Footer
    gr.HTML("""
        <hr style='margin: 2rem 0; border: none; border-top: 2px solid #e2e8f0;'>
        <div style='text-align: center; color: #64748b; padding: 1rem;'>
            <p style='margin: 0.5rem 0;'><strong style='color: #1e3a8a;'>Aircraft Threat Detection System</strong></p>
            <p style='margin: 0.5rem 0;'>CS 521 - Computer Vision | University of San Diego</p>
            <p style='margin: 0.5rem 0; font-size: 0.9rem;'>Powered by YOLOv8s Deep Learning Model</p>
        </div>
    """)

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
