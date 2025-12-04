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
        bool: True if threat (military aircraft), False otherwise (commercial)
    
    Note: The FGVC dataset contains military aircraft that were originally labeled
    as commercial. This function correctly identifies them as military threats
    based on aircraft type keywords, regardless of original dataset labels.
    """
    # Military aircraft keywords - these are threats
    # Comprehensive list covering all 195 classes in our dataset
    military_keywords = [
        # US/NATO Fighters
        'F-', 'F16', 'F18', 'F22', 'F35', 'F14', 'F15', 'F4', 'F2', 'F117',
        'F-16A', 'F/A-18',  # FGVC format
        
        # Bombers (US, Russian, Chinese)
        'B-', 'B1', 'B2', 'B21', 'B52',
        'Tu160', 'Tu22M', 'Tu95', 'Tu-160', 'Tu-22', 'Tu-95',  # Russian bombers
        'H6',  # Chinese bomber
        
        # Attack aircraft
        'A10', 'A-10', 'AV8B', 'A-7',
        'EMB314',  # Super Tucano (attack/trainer)
        
        # Attack helicopters
        'AH', 'AH64', 'Mi24', 'Mi28', 'Ka52', 'WZ', 'Z10', 'Z19',
        'WZ10', 'WZ7', 'WZ9',  # Chinese attack helicopters
        
        # Military helicopters
        'CH', 'UH', 'CH47', 'CH53', 'UH60', 'Mi8', 'Mi26', 'Ka27', 'V22', 'V280',
        
        # Russian aircraft
        'Mi', 'Su', 'Mig',
        'Su24', 'Su25', 'Su34', 'Su47', 'Su57',
        'Mig29', 'Mig31',
        'Il-76', 'Il76',  # Russian transport
        
        # European fighters
        'Eurofighter', 'Rafale', 'Tornado', 'EF2000', 'JAS39',
        'Mirage', 'Mirage2000',  # French fighter
        
        # Asian fighters
        'J10', 'J20', 'J35', 'J36', 'J50', 'JF17', 'JH7', 'FCK1', 'KF21', 'KAAN',
        'Tejas',  # Indian fighter
        
        # Military transport
        'KC135', 'KC-135', 'C-130', 'C130', 'C17', 'C5', 'C1', 'C2', 'C390', 'A400M',
        'C-47',  # WWII transport
        'An-12', 'An72', 'An124', 'An22', 'An225',  # Antonov military transports
        'Y20',  # Chinese transport
        
        # Drones/UAVs
        'MQ', 'RQ', 'TB', 'TB001', 'TB2', 'XQ58',
        'MQ9', 'MQ25', 'RQ4',
        'AKINCI', 'Bayraktar',  # Turkish UAVs
        
        # Reconnaissance
        'SR71', 'U2', 'US2',
        
        # AWACS/Maritime patrol
        'E2', 'E7', 'KJ600', 'P3',
        
        # Historic military
        'Spitfire', 'Vulcan',
        
        # Experimental
        'X29', 'X32', 'XB70', 'YF23',
        
        # Trainers that are military
        'Hawk', 'Hawk T1',
    ]
    
    # Check if aircraft class contains any military keyword
    aircraft_upper = aircraft_class.upper()
    return any(keyword.upper() in aircraft_upper for keyword in military_keywords)


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

