"""
Utility functions for aircraft threat detection
"""

import cv2
import numpy as np
from pathlib import Path
from config import (
    CONFIDENCE_THRESHOLD,
    THREAT_LABEL,
    SAFE_LABEL,
    BOX_COLOR_THREAT,
    BOX_COLOR_SAFE,
    BOX_THICKNESS,
    TEXT_COLOR,
    FONT_SCALE,
    FONT_THICKNESS
)


def is_threat(aircraft_class):
    """
    Determine if an aircraft class is a threat
    
    Args:
        aircraft_class: Name of the aircraft class
    
    Returns:
        bool: True if threat, False otherwise
    """
    # Option 1: All aircraft are threats
    return True
    
    # Option 2: Only military aircraft are threats (uncomment to use)
    # military_keywords = [
    #     'F-', 'F16', 'F18', 'F22', 'F35',  # Fighters
    #     'B-', 'B1', 'B2', 'B52',  # Bombers
    #     'A-10', 'AH', 'AH64',  # Attack helicopters
    #     'CH', 'UH',  # Transport helicopters
    #     'Mi', 'Su', 'Mig',  # Russian aircraft
    #     'Eurofighter', 'Rafale', 'Tornado',  # European fighters
    #     'J10', 'J20', 'J35',  # Chinese fighters
    #     'KC135', 'C-130', 'C17', 'C5',  # Military transport
    #     'MQ', 'RQ', 'TB2',  # Drones
    #     'SR71', 'U2',  # Reconnaissance
    # ]
    # return any(keyword in aircraft_class for keyword in military_keywords)


def draw_detection(img, x1, y1, x2, y2, aircraft_type, confidence, is_threat_detected):
    """
    Draw bounding box and label on image
    
    Args:
        img: Image to draw on (BGR format)
        x1, y1, x2, y2: Bounding box coordinates
        aircraft_type: Name of aircraft class
        confidence: Detection confidence score
        is_threat_detected: Whether this is a threat
    
    Returns:
        Image with drawn detection
    """
    # Choose color based on threat status
    color = BOX_COLOR_THREAT if is_threat_detected else BOX_COLOR_SAFE
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    
    # Prepare label
    if is_threat_detected:
        label = f"{THREAT_LABEL}: {aircraft_type}"
    else:
        label = f"{SAFE_LABEL}: {aircraft_type}"
    
    label += f" ({confidence:.2f})"
    
    # Calculate text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
    )
    
    # Draw label background
    cv2.rectangle(
        img,
        (x1, y1 - text_height - baseline - 10),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        img,
        label,
        (x1, y1 - baseline - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS
    )
    
    return img


def draw_status_banner(img, threat_detected, num_detections):
    """
    Draw status banner at top of image
    
    Args:
        img: Image to draw on (BGR format)
        threat_detected: Whether any threats were detected
        num_detections: Number of aircraft detected
    
    Returns:
        Image with status banner
    """
    if threat_detected:
        status_text = f"THREAT DETECTED! ({num_detections} aircraft)"
        status_color = (0, 0, 255)  # Red
    else:
        status_text = f"No Threats ({num_detections} aircraft detected)"
        status_color = (0, 255, 0)  # Green
    
    # Draw status banner
    cv2.rectangle(img, (10, 10), (600, 70), status_color, -1)
    cv2.putText(
        img,
        status_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        TEXT_COLOR,
        3
    )
    
    return img

