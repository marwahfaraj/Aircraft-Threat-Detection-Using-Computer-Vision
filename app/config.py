"""
Configuration file for Aircraft Threat Detection App
"""

from pathlib import Path

# Project root (parent of app/)
PROJECT_ROOT = Path(__file__).parent.parent

# Model Configuration
# Using YOLOv8s (Small) Augmented model - better performance than YOLOv8n
MODEL_PATH = PROJECT_ROOT / "results/model_2_output/yolov8s_augmented/weights/best.pt"
# Alternative paths (will be checked in app.py if primary doesn't exist):
# - results/model_1_output/augmented_on_fly/weights/best.pt (YOLOv8n)
# - results/model_2_output/yolov8s_baseline2/weights/best.pt (YOLOv8s baseline)
# - results/models_output/augmented_on_fly/weights/best.pt (old location)

CLASSES_FILE = PROJECT_ROOT / "data/processed/yolo_format/classes.txt"
CLASS_MAPPING_FILE = PROJECT_ROOT / "data/processed/yolo_format/class_mapping.json"

# Detection Configuration
CONFIDENCE_THRESHOLD = (
    0.35  # Minimum confidence for detection (increased for stability)
)
IOU_THRESHOLD = 0.45  # IoU threshold for NMS

# Threat Classification
THREAT_LABEL = "THREAT DETECTED"
SAFE_LABEL = "AIRCRAFT DETECTED"

# Visualization
BOX_COLOR_THREAT = (0, 0, 255)  # Red (BGR)
BOX_COLOR_SAFE = (0, 255, 255)  # Yellow (BGR)
BOX_THICKNESS = 3
TEXT_COLOR = (255, 255, 255)  # White
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Gradio Configuration
GRADIO_SERVER_NAME = "127.0.0.1"  # Use localhost instead of 0.0.0.0
GRADIO_SERVER_PORT = 7860  # Change to 7861, 7862, etc. if port is busy
GRADIO_SHARE = False  # Set to True for public link
