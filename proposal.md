# Image_Based_Classification_of_Residential_Exterior_Conditions
**Course:** MSML640 – Spring 2026  
**Author:** Emily Castelan Moreno  

---

# Problem Statement  

Homebuyers, renters, appraisers, and policymakers often rely on visual impressions of residential properties to assess quality, safety, and investment potential. However, these judgments are subjective and inconsistent. This project aims to explore whether a computer vision model can systematically classify the condition of residential properties based solely on exterior imagery.

Specifically, this project will develop an image classification model that categorizes residential properties into condition-based classes using visual cues such as building exterior quality, yard maintenance, driveway condition, and immediate surroundings.

---

# Objective  

The goal is to evaluate how well a pretrained deep learning model can be fine-tuned to classify residential exterior conditions and to analyze how data augmentation and synthetic data affect model performance and robustness.

---

# Dataset  

## Data Source  
The dataset will be constructed using a combination of:
- Manually collected images from **Google Street View**
- Publicly available images from online sources (e.g., Google Images, forums)
- Personal photographs where applicable  

All data will be collected manually (no automated scraping) to comply with platform policies.

---

## Classes  

Images will be labeled into three categories based on a structured scoring rubric:

1. **Poor Condition**
2. **Average Condition**
3. **Well-Maintained Condition**

---

## Labeling Methodology  

Each image will be scored based on four visual criteria:

- **House Exterior Condition** (paint, structure, roof, visible damage)  
- **Yard / Landscaping Condition**  
- **Driveway / Walkway Condition**  
- **Immediate Street Condition**  

Each category will be scored from 0–2:

- 0 = Poor  
- 1 = Average  
- 2 = Good  

Total score determines class label:
- 0–3 → Poor  
- 4–6 → Average  
- 7–8 → Well-Maintained  

This structured rubric ensures consistent and reproducible labeling.

---

## Dataset Size and Split  

The dataset will contain approximately:
- 150–300 total images  
- Balanced across the three classes  

Split:
- Training set: 70%  
- Validation set: 15%  
- Test set: 15%  

---

# Model Selection  

A pretrained convolutional neural network (CNN) will be used as the backbone model. Candidate models include:

- ResNet-50  
- MobileNetV2  
- EfficientNet-B0  

The final selection will be justified based on:
- Model complexity  
- Performance on small datasets  
- Computational efficiency  

---

# Training Configurations  

The model will be evaluated under four configurations:

1. **Baseline**  
   - Original dataset only  
   - No augmentation or synthesis  

2. **Data Augmentation**  
   - Apply transformations such as:
     - Rotation  
     - Cropping  
     - Brightness/contrast changes  
     - Horizontal flipping  

3. **Synthetic Data**  
   - Generate modified images using:
     - Noise injection  
     - Blur  
     - Lighting variation  

4. **Augmentation + Synthetic Data**  
   - Combination of both methods  

---

# Evaluation Plan  

For each configuration, the following will be reported:

- Training and validation loss curves  
- Accuracy per epoch  
- Confusion matrix on the test set  

---

## Comparative Analysis  

A detailed comparison will be conducted between:
- Baseline model  
- Best-performing configuration  

Analysis will include:
- Misclassification patterns  
- Class confusion (e.g., Average vs Well-Maintained)  
- Impact of visual ambiguity  

---

## Robustness Testing  

The best-performing model will be tested under:
- Image blur  
- Noise  
- Lighting changes  
- Partial occlusion  

This will evaluate how well the model generalizes to real-world variability.

---

# Expected Challenges  

- Ambiguity between “Average” and “Well-Maintained” classes  
- Variability in image framing and lighting  
- Potential bias from background or neighborhood features  

These challenges will be analyzed as part of the error analysis.

---

# Real-World Application  

This project has potential applications in:
- Real estate analysis  
- Urban planning and housing condition monitoring  
- Automated property assessment tools  

---

# Summary  

This project will implement a transfer learning-based image classification system to evaluate residential exterior conditions using real-world data. The focus will be on understanding model behavior, analyzing failure cases, and assessing the impact of data augmentation and synthetic data on generalization.

---