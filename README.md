# VetCyto Model Diagnostic Analysis - Cross-Domain Transfer Study

## Project Overview

This repository contains diagnostic code and analysis for investigating the failure of a YOLOv8 object detection model trained on optical microscopy images when applied to holographic microscopy images. 
The project demonstrates systematic model diagnosis using Explainable AI (XAI) techniques to identify learned biases and failure modes.

## Research Context

**Problem Statement:**  
A YOLOv8 model trained for cell detection in optical microscopy images completely fails when applied to holographic microscopy images (0% detection rate). 
Through preliminary XAI analysis, we discovered the model learned spurious correlations with tape border artifacts in the training data rather than actual cellular morphology.

**Research Questions:**
1. What features does the model actually rely on for detection?
2. Why does the model fail to detect large cellular structures even in optical images?
3. Why does image sharpening appear to "enable" detection in holographic images?
4. Can we quantify the domain shift between optical and holographic imaging modalities?


```
```

## Experimental Results Summary

### Optical Microscopy (Training Domain)
- **Total Detections:** 3,994
- **Success Rate:** 85.0% (345/406 patches)
- **Avg Detections per Patch:** 9.84
- **Avg Confidence:** 0.4424
- **Issues Identified:**
  - ✅ Small cells correctly detected
  - ❌ Large cellular structures missed
  - ❌ Tape borders falsely detected as objects

### Holographic Microscopy (Target Domain)

#### Baseline (No Preprocessing)
- **Total Detections:** 0
- **Success Rate:** 0.0% (0/325 patches)
- **Avg Confidence:** 0.0000
- **Conclusion:** Model completely rejects holographic images

#### After Image Sharpening

| Configuration | Detections | Success Rate | Interpretation |
|--------------|------------|--------------|----------------|
| Sharpen ×1 | 1,509 | 82.1% | Artificial edge creation |
| Sharpen ×2 | 13,480 | 90.4% | Increased artifacts |
| Sharpen ×3 | 36,537 | 90.4% | Massive false positives |

**Critical Finding:** Sharpening does NOT enable genuine cell detection. Instead, it creates artificial high-contrast edges that match the model's learned tape border features, resulting in false positive detections on noise and background texture.

## XAI Methods Implemented

### 1. Gradient-weighted Class Activation Mapping (Grad-CAM)
**Purpose:** Visualize which spatial regions the model attends to when making predictions.

**Implementation:**
```python
from pytorch_grad_cam import GradCAM
cam = GradCAM(model=model.model, target_layers=[last_conv_layer])
heatmap = cam(input_tensor=image)
```

**Expected Output:** Heatmaps showing attention on actual cells vs. borders/artifacts.

### 2. SHAP (SHapley Additive exPlanations)
**Purpose:** Quantify the contribution of individual pixels to detection scores.

**Implementation:**
```python
import shap
explainer = shap.KernelExplainer(model_predict, background)
shap_values = explainer.shap_values(image)
```

**Expected Output:** Pixel attribution maps (red = positive contribution, blue = negative).

### 3. Occlusion Sensitivity Analysis
**Purpose:** Systematically mask image regions to identify critical features.

**Implementation:**
```python
# Iteratively occlude regions and measure detection drop
for region in image_regions:
    occluded_image = mask_region(image, region)
    sensitivity[region] = baseline_detections - occluded_detections
```

**Expected Output:** Sensitivity maps showing which regions are critical for detections.

### 4. Detection Size Distribution Analysis
**Purpose:** Understand what object sizes and shapes the model learned.

**Implementation:**
```python
# Analyze all bounding boxes
box_areas = [(x2-x1) * (y2-y1) for (x1,y1,x2,y2) in detections]
aspect_ratios = [min(w,h) / max(w,h) for w,h in box_dimensions]
```

**Expected Output:** Histograms showing size constraints and shape biases.

## Key Diagnostic Findings

### 1. Tape Border Artifact Bias (Confirmed)
- **Evidence:** Grad-CAM shows high attention on rectangular frame borders
- **Evidence:** SHAP analysis shows positive attribution for border pixels
- **Evidence:** Occlusion of borders significantly reduces detection counts
- **Conclusion:** Model learned spurious correlation with tape artifacts

### 2. Size Constraint Bias
- **Evidence:** Maximum detected area << large blob area
- **Evidence:** No detections on structures >10,000 pixels²
- **Conclusion:** Training data contained only small cells; model cannot generalize to large structures

### 3. Domain Shift Analysis
**Optical vs. Holographic Characteristics:**

| Feature | Optical | Holographic |
|---------|---------|-------------|
| Contrast | High (40-70 std) | Low (15-30 std) |
| Edge Density | 0.10-0.20 | 0.02-0.05 |
| Brightness | 120-180 | 60-100 |
| Texture | Clear boundaries | Interference patterns |
| Tape Borders | Present | Absent |

**Conclusion:** Fundamental domain mismatch prevents transfer learning without target domain training data.

## Running the Diagnostic Suite

### Prerequisites
```bash
pip install ultralytics torch opencv-python numpy pandas matplotlib
pip install grad-cam shap scikit-learn
```

### Step 1: Baseline Analysis (Optical Images)
```bash
python diagnostic_scripts/01_optical_baseline_analysis.py
```
**Inputs:** 
- YOLOv8 model (.pt file)
- Optical microscopy image

**Outputs:**
- `results/optical/full_image_baseline.png`
- `results/optical/patch_detections/`
- `results/optical/summary_statistics.csv`

### Step 2: Baseline Analysis (Holographic Images)
```bash
python diagnostic_scripts/02_holographic_baseline_analysis.py
```
**Expected Result:** 0 detections (confirms domain shift)

### Step 3: Sharpening Experiments (Holographic)
```bash
python diagnostic_scripts/03_sharpening_experiments.py
```
**Expected Result:** False positive explosion (confirms artifact detection)

### Step 4: XAI Diagnostic Suite
```bash
python diagnostic_scripts/04_xai_diagnostic_suite.py
```
**Outputs:**
- `results/xai_analysis/gradcam_*.png` (attention heatmaps)
- `results/xai_analysis/shap_*.png` (pixel attribution)
- `results/xai_analysis/occlusion_sensitivity_*.png` (feature importance)
- `results/xai_analysis/detection_size_analysis.png` (size distribution)

## Interpretation Guide

### Grad-CAM Heatmaps
- **Red/Yellow regions:** High model attention
- **Blue/Green regions:** Low model attention
- **Diagnostic Question:** Does attention focus on cells or borders?

### SHAP Attribution Maps
- **Red pixels:** Increase detection confidence
- **Blue pixels:** Decrease detection confidence
- **Diagnostic Question:** Do border pixels have high positive attribution?

### Occlusion Sensitivity
- **Bright areas:** Critical for detections (removing them kills detections)
- **Dark areas:** Not important for detections
- **Diagnostic Question:** Are borders critical while cells are not?

### Size Distribution
- **Narrow distribution:** Model learned specific size constraints
- **Elongated shapes (aspect ratio < 0.6):** Model detects edges/borders
- **Circular shapes (aspect ratio > 0.7):** Model detects round cells

## Diagnostic Conclusions

### Why Model Fails on Holographic Images

1. **Primary Cause:** Model learned tape border artifacts as discriminative features during training
2. **Secondary Cause:** Domain shift in image characteristics (contrast, texture, edge density)
3. **Tertiary Cause:** Absence of tape borders in holographic images removes primary learned feature

### Why Sharpening "Appears" to Work

1. Sharpening creates artificial high-contrast edges
2. These edges match the tape border features the model learned
3. Result: Model detects sharpening artifacts, not actual cells
4. **Evidence:** Visual inspection shows detections on empty background, not on pollen grains

### Why Model Misses Large Structures (Even in Optical Images)

1. Training data contained only small cells (area < 5,000 pixels²)
2. Model learned implicit size constraints
3. Large structures fall outside learned distribution
4. **Evidence:** No detections with area > 10,000 pixels² in entire dataset

## Recommendations

### For Cross-Domain Transfer
**❌ Do NOT use image enhancement (sharpening) as a solution**
- Creates false positives
- Does not address root cause
- Not scientifically valid

**✅ Recommended Approach:**
1. Annotate 100-200 holographic microscopy images
2. Fine-tune YOLOv8 on target domain using transfer learning
3. Freeze early layers, train detection head on new data
4. Validate with proper ground truth annotations

### For Model Retraining
**✅ Data Collection Best Practices:**
1. Remove tape borders from training images (or mask them out)
2. Include diverse cell sizes (small, medium, large)
3. Ensure balanced representation of cell morphologies
4. Use data augmentation (rotation, scaling, brightness) during training

### For Future Work
1. **Quantitative Validation:** Manually annotate 50 holographic patches for precision/recall calculation
2. **Classical CV Baseline:** Implement morphological detection for comparison
3. **Multi-Domain Training:** Train on both optical AND holographic data simultaneously
4. **Architecture Modifications:** Explore attention mechanisms that focus on cell morphology

#
```
---

