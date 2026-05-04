# Image-Based Classification of Residential Exterior Conditions
**Course:** MSML640 – Spring 2026  
**Author:** Emily Castelan Moreno  

---

## Overview

This project investigates whether a deep learning model can classify the condition of residential properties using only exterior images. The model is trained using transfer learning and evaluated under multiple data configurations to understand how data augmentation and synthetic transformations affect performance.

---

## Problem Statement

Assessing residential property condition is often subjective and inconsistent. This project aims to determine whether a computer vision model can systematically classify properties into condition-based categories using visual features such as building exterior quality, landscaping, and surrounding environment.

---

## Dataset

The dataset consists of manually collected images from:
- Google Street View

Each image is labeled into one of three classes:
- Poor
- Average
- Well-maintained

Labels are assigned using a structured scoring rubric based on:
- House exterior condition
- Yard and landscaping
- Driveway and walkway condition
- Immediate street condition

---
## HOW TO RUN:  



---

## Project Structure
```
Image-based Classification of Residential Exterior Conditions/
├── data/
│   ├── house_images/        # Original labeled images (by class)
│   │   ├── poor/
│   │   ├── average/
│   │   └── well_maintained/
│   │
│   ├── train/               # Training split
│   │   ├── poor/
│   │   ├── average/
│   │   └── well_maintained/
│   │
│   ├── val/                 # Validation split
│   │   ├── poor/
│   │   ├── average/
│   │   └── well_maintained/
│   │
│   ├── test/                # Test split
│   │   ├── poor/
│   │   ├── average/
│   │   └── well_maintained/
│   │
│   └── labels.csv           # Ground truth labels
│
├── src/
│   ├── data_split.py        # Verifies dataset and creates train/val/test splits
│   ├── dataset.py           # Data loading and transformations
│   ├── model.py             # Pretrained model definition (MobileNetV2)
│   ├── train.py             # Training pipeline
│   └── evaluate.py          # (Optional) Evaluation and metrics
│
├── notebooks/
│   └── data_cleaning.ipynb  # One-time data preprocessing and fixes
│
├── outputs/                 # Generated during experiments
│   ├── models/              # Saved model weights
│   ├── plots/               # Loss/accuracy curves
│   └── confusion_matrices/  # Evaluation results
```
---

## Model  
The project uses MobileNetV2 pretrained on ImageNet. The feature extraction layers are frozen, and the final classification layer is replaced to output three classes.  
This approach is chosen to:  
- Reduce overfitting on a small dataset
- Leverage pretrained visual features
- Enable efficient training

---

## Training Configurations

The model is evaluated under four configurations:

1. Baseline  
   Original dataset without modifications  

2. Data Augmentation  
   Includes transformations such as rotation, flipping, and brightness changes  

3. Synthetic Data  
   Includes noise, blur, and lighting variations  

4. Augmentation + Synthetic  
   Combination of both methods  


---

## Evaluation

For each configuration, the following are analyzed:  
- Training and validation loss curves
- Accuracy over epochs
- Confusion matrix on the test set
A comparative analysis is conducted between the baseline and the best-performing configuration.

---
## Robustness Testing

The final model is evaluated under:
- Image blur
- Noise
- Lighting variation
- Partial occlusion
This tests how well the model generalizes to real-world variability.