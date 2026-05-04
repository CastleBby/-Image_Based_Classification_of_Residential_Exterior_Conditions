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
1. clone the repo
```
git clone https://github.com/CastleBby/-Image_Based_Classification_of_Residential_Exterior_Conditions.git
```
2. cd to root 
```
cd -Image_Based_Classification_of_Residential_Exterior_Conditions
```
3. activate the venv 
```
source venv/bin/activate
```
4. install dependencies 
```
pip install -r requirements.txt
```
5. run the data_split.py 
```
python -m src.data_split
```
6. generate the synthetic and combined data
```
python -m src.generate_data
```
7. train the model on each configurations 
```
python -m src.train --config baseline --epochs 20
python -m src.train --config augment --epochs 20
python -m src.train --config synthetic --epochs 20
python -m src.train --config combined --epochs 20
```
8. go through the ipynb to evaluate the model and test the robustness
- test accuracy 
- confusion matrices 
- robustness results
9. run the frontend demo 
```
streamlit run app.py
```

---

## Project Structure
```
Image-based Classification of Residential Exterior Conditions/
project/
├── data/
│   ├── house_images/
│   |    ├── train/
│   |    ├── val/
│   |    └── test/
│   └── house_labels.csv
│
├── src/
│   ├── data_split.py        # dataset verification + splitting
│   ├── dataset.py           # dataloaders + transforms
|   ├── grading_rubric.py    # house scoring method in python
│   ├── model.py             # model definition
│   ├── train.py             # training + evaluation loop
│
├── notebooks/
│   └── data_cleaning.ipynb  # one time use for cleanign raw data
│
├── outputs/
│   ├── models/
│   ├── plots/
│   └── results/
│
├── README.md
├── proposal.md
├── proposal.pdf
├── grading_rubric.md         # used to score houses
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