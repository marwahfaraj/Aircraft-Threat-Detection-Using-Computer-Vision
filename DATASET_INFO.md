# Dataset Information

This document provides a quick reference for the datasets included in this project.

## 📊 Available Datasets

### 1. Military Aircraft Detection Dataset
**Location:** `data/dataset/` and `data/labels_with_split.csv`

**Statistics:**
- Total Images: 22,177
- Training Split: 3,335 images
- Validation Split: 3,334 images  
- Test Split: 3,334 images
- Total Annotations: 39,483 (multiple objects per image)

**Format:**
- Images: `data/dataset/{filename}.jpg`
- Annotations: CSV format with columns:
  - `filename`: Image filename (without extension)
  - `width`, `height`: Image dimensions
  - `class`: Aircraft type (Mi28, F16, UH60, etc.)
  - `xmin`, `ymin`, `xmax`, `ymax`: Bounding box coordinates
  - `split`: train/val/test designation

**Use Case:**
- Object detection training (YOLO)
- Military aircraft identification
- Threat detection and localization

### 2. FGVC-Aircraft 2013b Dataset
**Location:** `data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/`

**Statistics:**
- Total Images: 10,000
- Training Split: ~3,333 images
- Validation Split: ~3,333 images
- Test Split: ~3,334 images
- Unique Variants: 102
- Unique Families: 70
- Unique Manufacturers: 41

**Format:**
- Images: `data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/{image_id}.jpg`
- Annotations: Text files
  - `images_variant_train.txt`, `images_variant_val.txt`, `images_variant_test.txt`
  - `images_family_train.txt`, etc.
  - `images_manufacturer_train.txt`, etc.
  - `images_box.txt`: Bounding boxes for all images

**Use Case:**
- Fine-grained aircraft classification
- Variant identification (Boeing 737-700, F-16, etc.)
- Commercial vs. military classification
- Transfer learning for improved accuracy

## 🎯 Recommended Usage Strategy

### Phase 1: Object Detection
Use the **Military Aircraft Detection Dataset** to:
1. Train YOLO model for aircraft localization
2. Detect and identify military aircraft types
3. Generate bounding boxes for aircraft in images

### Phase 2: Classification
Use the **FGVC-Aircraft Dataset** to:
1. Train CNN/ResNet for fine-grained classification
2. Identify specific aircraft variants
3. Distinguish between commercial and military aircraft

### Phase 3: Integration
Combine both datasets to create a comprehensive threat detection system:
1. Detect aircraft using YOLO (from Dataset 1)
2. Classify detected aircraft into specific variants (from Dataset 2)
3. Assess threat level:
   - **High threat**: Military aircraft (fighters, bombers, attack helicopters)
   - **Low threat**: Commercial aircraft (passenger planes, cargo planes)
   - **Unknown**: Unidentified or unclear aircraft

## 📝 Data Loading Examples

### Load Military Aircraft Data
```python
import pandas as pd
from pathlib import Path

# Load annotations
train_df = pd.read_csv('data/train.csv')
labels_df = pd.read_csv('data/labels_with_split.csv')

# Access image
image_path = Path('data/dataset') / f"{filename}.jpg"
```

### Load FGVC Aircraft Data
```python
# Load variant annotations
def load_fgvc_annotations(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                image_id, label = parts
                data.append({'image_id': image_id, 'label': label})
    return pd.DataFrame(data)

fgvc_train = load_fgvc_annotations('data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images_variant_train.txt')
```

## 🔍 Explore the Data

Run the Jupyter notebook to explore both datasets:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

The notebook includes:
- Dataset statistics and distributions
- Sample visualizations with bounding boxes
- Class distribution analysis
- Dataset comparison
- Next steps for model training

## 📚 References

- **Military Aircraft Dataset**: Various Kaggle/public sources
- **FGVC-Aircraft**: [Official Paper](http://arxiv.org/abs/1306.5151)
  - Maji, S., Kannala, J., Rahtu, E., Blaschko, M., & Vedaldi, A. (2013). Fine-Grained Visual Classification of Aircraft.

## ⚠️ Note

Make sure the data directory is included in `.gitignore` to avoid pushing large image files to GitHub. Only share code, notebooks, and documentation with your team.

