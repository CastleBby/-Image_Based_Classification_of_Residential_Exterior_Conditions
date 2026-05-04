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

## Project Structure


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