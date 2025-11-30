# Video Presentation Outline
## Aircraft Threat Detection Using Computer Vision

**Duration:** 10-12 minutes
**Format:** Recorded video with slides and narration

---

## SLIDE 1: Cover Slide (0:00-0:30)

### Content
- **Title:** Aircraft Threat Detection Using Computer Vision
- **Subtitle:** Real-Time Detection and Classification Using YOLOv8
- **Team Number:** [Your Team Number]
- **Team Members:** [All names]
- **Course:** CS 521 - Computer Vision
- **Institution:** University of San Diego
- **Date:** [Presentation Date]

### Narration (Speaker 1)
"Good [morning/afternoon]. Welcome to our presentation on Aircraft Threat Detection Using Computer Vision. I'm [Name], and joining me today are my teammates [Names]. Today, we'll present our final project for CS 521, where we developed a real-time aircraft detection system capable of identifying and classifying aircraft threats."

---

## SLIDE 2: Problem Statement & Motivation (0:30-1:30)

### Content
- **The Challenge:**
  - Manual aircraft monitoring is labor-intensive and error-prone
  - Cannot scale to modern airspace traffic volumes
  - Critical for defense, security, and aerospace monitoring
  
- **Our Solution:**
  - Automated computer vision system
  - Real-time detection and classification
  - 195 aircraft types (commercial + military)

### Visuals
- Image of crowded airspace
- Security monitoring scenario
- Problem → Solution diagram

### Narration (Speaker 1)
"The detection and classification of aircraft in real-time is crucial for aerospace monitoring, defense systems, and security operations. Traditional manual methods cannot handle modern airspace volumes and are prone to human error. Our project addresses this challenge by developing an automated computer vision system that can detect and classify aircraft in real-time, distinguishing between 195 different aircraft types including both commercial and military variants."

---

## SLIDE 3: Dataset Overview (1:30-2:30)

### Content
- **Two Major Datasets Combined:**
  1. Military Aircraft Detection (39,482 images, 96 types)
  2. FGVC-Aircraft 2013b (10,000 images, 100 variants)
  
- **Final Dataset Statistics:**
  - Total: 49,482 labeled images
  - Classes: 195 unique aircraft types
  - Splits: 70% train / 18% val / 12% test
  - Military: 79.8% | Commercial: 20.2%

### Visuals
- Sample aircraft images (grid format)
- Pie chart showing military vs. commercial distribution
- Bar chart of class distribution

### Narration (Speaker 2)
"Our dataset combines two publicly available aircraft datasets from Kaggle. The Military Aircraft Detection dataset provides 39,482 images covering 96 military aircraft types, while the FGVC-Aircraft dataset contributes 10,000 images of 100 commercial aircraft variants. Combined, this gives us 49,482 labeled images spanning 195 unique aircraft classes. We split this into 70% training, 18% validation, and 12% test sets. The dataset is imbalanced, with military aircraft comprising about 80% of the data, which reflects realistic monitoring scenarios."

---

## SLIDE 4: Sample Images from Dataset (2:30-3:00)

### Content
- Grid of diverse aircraft images showing:
  - Commercial jets (Boeing 737, 777, Airbus A320, A380)
  - Military fighters (F-16, F-22, F-35)
  - Bombers (B-52, B-2)
  - Transport aircraft (C-130, C-17)
  - Helicopters (AH-64, UH-60)

### Visuals
- 3×4 grid of representative images
- Labels showing aircraft type and category

### Narration (Speaker 2)
"Here we see representative samples from our dataset, showcasing the diversity in aircraft types, viewing angles, and imaging conditions. This variety presents both challenges and opportunities for developing a robust detection system."

---

## SLIDE 5: Methodology - Preprocessing (3:00-4:00)

### Content
- **Data Processing Pipeline:**
  1. Dataset integration and standardization
  2. Image resizing (640×640 pixels)
  3. YOLO format conversion
  4. Quality control and validation
  5. Train/val/test splitting

- **Data Augmentation:**
  - Geometric: Translation, scaling, flipping, mosaic
  - Color: HSV adjustments (hue, saturation, value)
  - Advanced: Mixup, copy-paste

### Visuals
- Flowchart of preprocessing pipeline
- Before/after augmentation examples
- Split into 2 columns: Original → Augmented

### Narration (Speaker 3)
"Our methodology began with extensive data preprocessing. We standardized both datasets into YOLO format, resized images to 640×640 pixels, and performed quality control to remove corrupted files. To improve model generalization, we implemented comprehensive data augmentation including geometric transformations like translation and scaling, color space adjustments, and advanced techniques like mosaic augmentation which combines four images into one."

---

## SLIDE 6: Model Architectures (4:00-4:45)

### Content
- **YOLOv8 Architecture:**
  - State-of-the-art object detection
  - Single-stage detector (fast & accurate)
  - Anchor-free detection head
  
- **Two Model Variants:**
  
  | Model | Parameters | Size | Speed |
  |-------|-----------|------|--------|
  | YOLOv8n | 3.2M | 6 MB | 476 FPS |
  | YOLOv8s | 11.2M | 22 MB | 263 FPS |

### Visuals
- YOLOv8 architecture diagram
- Comparison table
- Model size visualization

### Narration (Speaker 3)
"We evaluated two YOLOv8 model variants. YOLOv8 is a state-of-the-art single-stage object detector known for its excellent balance of speed and accuracy. We tested YOLOv8-Nano, a lightweight 3.2 million parameter model, and YOLOv8-Small with 11.2 million parameters. Both models achieve real-time performance, with the Nano variant processing 476 frames per second and the Small variant achieving 263 FPS on GPU."

---

## SLIDE 7: Training Configuration (4:45-5:30)

### Content
- **Training Setup:**
  - Hardware: Google Colab Pro (NVIDIA A100)
  - Framework: Ultralytics YOLOv8 (PyTorch)
  - Epochs: 50 with early stopping
  - Batch size: 16
  - Optimizer: SGD with cosine annealing
  
- **Four Models Trained:**
  1. YOLOv8n Baseline (no augmentation)
  2. YOLOv8n Augmented
  3. YOLOv8s Baseline
  4. YOLOv8s Augmented

### Visuals
- Training configuration table
- Hardware setup image
- Timeline showing ~6-8 hours per model

### Narration (Speaker 4)
"We trained four models: baseline and augmented versions of both YOLOv8n and YOLOv8s. Training was conducted on Google Colab Pro using NVIDIA A100 GPUs. Each model was trained for up to 50 epochs with early stopping, taking approximately 6 to 8 hours. We used stochastic gradient descent with cosine annealing for the learning rate schedule."

---

## SLIDE 8: Training Curves (5:30-6:15)

### Content
- **Learning Curves Showing:**
  - Training and validation loss
  - mAP@0.5 progression
  - Precision and recall evolution
  - Comparison: Baseline vs. Augmented

### Visuals
- 2×2 grid of training curves:
  - Top left: YOLOv8n training loss
  - Top right: YOLOv8n mAP@0.5
  - Bottom left: YOLOv8s training loss
  - Bottom right: YOLOv8s mAP@0.5
- Highlight convergence and improvements

### Narration (Speaker 4)
"These training curves show the learning progress of our models. Notice how the augmented models achieve better validation performance despite slightly higher training loss. This indicates that augmentation successfully prevented overfitting and improved generalization. All models converged smoothly without significant fluctuations."

---

## SLIDE 9: Results - Performance Comparison (6:15-7:30)

### Content
**Test Set Performance Table:**

| Model | Variant | mAP@0.5 | Precision | Recall |
|-------|---------|---------|-----------|--------|
| YOLOv8n | Baseline | 37.7% | 35.8% | 39.0% |
| YOLOv8n | Augmented | **53.1%** | 47.1% | 53.2% |
| YOLOv8s | Baseline | 45.1% | 50.4% | 38.4% |
| YOLOv8s | Augmented | **67.1%** | 64.9% | 63.6% |

**Key Findings:**
- 🎯 Best Model: YOLOv8s Augmented (67.1% mAP@0.5)
- 📈 Augmentation Impact: +41% to +49% improvement
- 🚀 Model Size Impact: +26% improvement (YOLOv8s vs YOLOv8n)

### Visuals
- Bar chart comparing all four models
- Highlight best performer
- Show improvement percentages

### Narration (Speaker 1)
"Here are our key results. On the test set, YOLOv8s with augmentation achieved the best performance at 67.1% mean Average Precision. This represents a 49% improvement over the baseline and a 26% improvement over the smaller YOLOv8n model. Data augmentation dramatically improved all metrics, with gains of 41% to 49% across both model sizes. The larger YOLOv8s model better captured the variety and complexity of our 195 aircraft classes."

---

## SLIDE 10: Confusion Matrix Analysis (7:30-8:00)

### Content
- **Confusion Matrix Highlights:**
  - Strong diagonal (good per-class accuracy)
  - Common confusions between similar variants:
    - Boeing 737-700 ↔ 737-800
    - Airbus A320 family variants
    - F-15 ↔ F-16
  
- **Performance by Category:**
  - High (>80% mAP): Large distinctive aircraft (747, A380, B-52)
  - Medium (50-80%): Common variants (737 family, F-16)
  - Challenging (<50%): Similar variants with limited data

### Visuals
- Confusion matrix visualization
- Zoom-in on problematic regions
- Sample confused aircraft pairs

### Narration (Speaker 2)
"Our confusion matrix analysis reveals interesting patterns. The strong diagonal indicates good overall performance, but we see confusion between visually similar aircraft variants. For example, Boeing 737 variants frequently confused each other, as do Airbus A320 family members. This aligns with human recognition challenges – these aircraft look nearly identical even to trained observers."

---

## SLIDE 11: Detection Examples (8:00-8:30)

### Content
- **Successful Detections:**
  - 4-6 sample images with bounding boxes
  - Show correct classifications
  - Various aircraft types and conditions
  - Confidence scores displayed

### Visuals
- Grid of successful detection examples
- Green bounding boxes
- Labels showing aircraft type and confidence

### Narration (Speaker 2)
"Here are examples of successful detections from our test set. Notice the accurate bounding boxes tightly fitting each aircraft, correct classifications, and high confidence scores. Our model handles various viewing angles, scales, and imaging conditions effectively."

---

## SLIDE 12: Deployment Application (8:30-9:30)

### Content
- **Real-Time Web Application:**
  - Built with Gradio framework
  - Features:
    - Live webcam detection
    - Image upload option
    - Automatic threat classification
    - Color-coded bounding boxes (Red=Threat, Yellow=Safe)
    - Confidence scores and aircraft labels

### Visuals
- Screenshot of Gradio interface
- Architecture diagram
- Feature highlights with annotations

### Narration (Speaker 3)
"To demonstrate practical applicability, we deployed our best model in a real-time web application using Gradio. The application features live webcam detection, image upload capability, and automatic threat classification. Detected military aircraft are labeled as threats with red bounding boxes, while commercial aircraft are marked as safe with yellow boxes. The system displays confidence scores and aircraft types for each detection."

---

## SLIDE 13: Live Demo (9:30-10:00)

### Content
- **Screen Recording or Live Demo:**
  - Show application launching
  - Upload sample image → detection result
  - Show multiple detections
  - Demonstrate threat classification
  - Show real-time performance

### Visuals
- Screen recording of application in action
- Show both threat and safe detections
- Highlight key features

### Narration (Speaker 3)
"Let me show you a quick demonstration of our deployed system. [Describe what's happening in the demo: launching the app, uploading an image, seeing detections appear, pointing out threat labels, showing confidence scores]. As you can see, the system processes images quickly and provides clear, actionable results."

---

## SLIDE 14: Key Findings & Discussion (10:00-10:45)

### Content
**Major Insights:**

1. **Data Augmentation is Critical**
   - +41% to +49% performance improvement
   - Essential for generalization

2. **Model Capacity Matters**
   - Larger models handle complexity better
   - +26% improvement with YOLOv8s

3. **Real-Time Performance Achieved**
   - 263-476 FPS on GPU
   - Suitable for practical deployment

4. **Challenges Remain:**
   - Fine-grained variant distinction
   - Small aircraft at distance
   - Class imbalance effects

### Visuals
- 4 key findings with icons
- Challenge examples

### Narration (Speaker 4)
"Our key findings highlight three critical insights. First, data augmentation is essential, providing 41% to 49% performance improvements. Second, model capacity matters – the larger YOLOv8s significantly outperformed the smaller variant. Third, we achieved true real-time performance suitable for practical deployment. However, challenges remain in distinguishing visually similar variants and detecting very small aircraft."

---

## SLIDE 15: Future Work (10:45-11:15)

### Content
**Next Steps for Improvement:**

- 🔧 **Model Enhancements:**
  - Experiment with larger YOLOv8 variants
  - Implement class-weighted loss functions
  - Try ensemble methods

- 📊 **Data Improvements:**
  - Collect more data for rare classes
  - Add video sequences for tracking
  - Include diverse conditions

- 🚀 **Deployment Extensions:**
  - Mobile application
  - Cloud-based API
  - Integration with security systems
  - Distance and altitude estimation

### Visuals
- Roadmap diagram
- Future features mockups

### Narration (Speaker 4)
"Future work includes several promising directions. We plan to experiment with even larger model variants and more sophisticated training techniques. Expanding the dataset with rare aircraft classes and video sequences would improve robustness. On the deployment side, we envision mobile applications, cloud-based APIs, and integration with existing security infrastructure."

---

## SLIDE 16: Conclusions (11:15-11:45)

### Content
**Project Summary:**

✅ Successfully developed aircraft threat detection system

✅ Achieved 67.1% mAP@0.5 on 195-class problem

✅ Demonstrated critical importance of data augmentation

✅ Deployed functional real-time web application

✅ Validated YOLOv8's effectiveness for aircraft detection

**Impact:**
- Proof-of-concept for automated airspace monitoring
- Foundation for practical security applications
- Demonstrates modern CV techniques on real-world problem

### Visuals
- Project highlights summary
- Key metrics recap
- Success indicators

### Narration (Speaker 1)
"In conclusion, we successfully developed a comprehensive aircraft threat detection system that achieves 67% mean Average Precision on a challenging 195-class problem. We validated the critical importance of data augmentation and demonstrated the system's real-world applicability through our deployed web application. This project serves as a strong proof-of-concept for automated aircraft monitoring and provides a foundation for future security applications."

---

## SLIDE 17: Teamwork & Contributions (11:45-12:00)

### Content
**Team Contributions:**

- **[Member 1 Name]:** 
  - Dataset collection and preprocessing
  - YOLOv8n model training
  - Report writing (Introduction, Methods)

- **[Member 2 Name]:**
  - Data augmentation implementation
  - YOLOv8s model training
  - Results analysis and visualization

- **[Member 3 Name]:**
  - Deployment application development
  - Code documentation
  - Report writing (Results, Discussion)

- **[Member 4 Name]:** (if applicable)
  - Testing and validation
  - Presentation creation
  - Report writing (Conclusion, Appendix)

**All members:** Collaborative problem-solving, team meetings, peer review

### Visuals
- Team member photos (optional)
- Contribution matrix
- Team workflow diagram

### Narration (All Members - rotate)
[Member 1]: "Our team worked collaboratively throughout this project. I focused on dataset collection and preprocessing, as well as training the YOLOv8n models."

[Member 2]: "I implemented the data augmentation pipeline and trained the YOLOv8s models, while also conducting detailed results analysis."

[Member 3]: "I developed the deployment application and ensured our code was well-documented and accessible."

[Member 4 if applicable]: "I handled testing, validation, and helped bring everything together for this presentation."

[All]: "We met regularly, divided work equitably, and reviewed each other's contributions to ensure quality."

---

## SLIDE 18: Thank You & Questions (12:00+)

### Content
- **Thank You!**
- **Questions?**
- **Contact Information** (optional)
- **GitHub Repository:** [link]
- **Acknowledgments:**
  - CS 521 course instructors
  - Dataset providers
  - Ultralytics YOLOv8 framework
  - Google Colab Pro

### Visuals
- Clean thank you slide
- Project logo/image
- GitHub QR code (optional)

### Narration (Speaker 1)
"Thank you for your attention. We're happy to answer any questions you may have. Our complete code, documentation, and results are available on our GitHub repository. We'd like to acknowledge our CS 521 instructors, the dataset providers, and the Ultralytics team for their excellent YOLOv8 framework."

---

## RECORDING TIPS

### Before Recording:
1. **Practice run-through** - Do at least 2 full rehearsals
2. **Check timing** - Ensure 10-12 minute target
3. **Test equipment** - Verify audio and video quality
4. **Close distractions** - Turn off notifications
5. **Good lighting** - Ensure faces are well-lit
6. **Stable internet** - If recording via Zoom/Teams

### During Recording:
1. **Speak clearly** - Moderate pace, enunciate
2. **Sound professional** - Avoid "um," "uh," filler words
3. **Look engaged** - Smile, make eye contact with camera
4. **Smooth transitions** - Hand off between speakers cleanly
5. **Point to slides** - Use cursor to highlight important items
6. **Show enthusiasm** - Be proud of your work!

### After Recording:
1. **Watch entire video** - Check quality
2. **Verify audio sync** - Ensure no delays
3. **Check timing** - Confirm 10-12 minutes
4. **Test file** - Ensure .mp4 plays on different devices
5. **Get team approval** - All members review before submission

---

## TECHNICAL SPECIFICATIONS

**Video Settings:**
- Resolution: 1920×1080 (1080p) or 1280×720 (720p minimum)
- Format: .mp4 (H.264 codec recommended)
- Frame Rate: 30 FPS
- Bitrate: 5-8 Mbps

**Audio Settings:**
- Sample Rate: 48 kHz
- Bitrate: 192 kbps
- Mono or stereo acceptable
- No background noise

**File Size:**
- Target: < 500 MB
- Maximum: Per Canvas limits (usually 500 MB - 1 GB)
- Compress if necessary using HandBrake or similar

---

**Good luck with your presentation!**

