# Residential Exterior Condition Scoring Rubric

## Overview
This rubric defines a structured and reproducible method for labeling residential exterior images into condition-based classes. Each image is evaluated across four categories, each scored from 0 to 2. The total score determines the final class label.

---

## Scoring Categories

### 1. House Exterior Condition (0–2)
Evaluates the visible condition of the structure itself.

- **0 (Poor)**  
  - Peeling or damaged paint  
  - Visible structural damage (cracks, broken siding, boarded windows)  
  - Roof deterioration or missing components  

- **1 (Average)**  
  - Minor wear and tear  
  - Slight discoloration or aging materials  
  - No major structural issues  

- **2 (Good)**  
  - Clean, intact exterior  
  - Recently maintained or renovated appearance  
  - No visible damage  

---

### 2. Yard / Landscaping Condition (0–2)
Evaluates maintenance of surrounding land.

- **0 (Poor)**  
  - Overgrown grass or weeds  
  - Dead vegetation or clutter  
  - Unmaintained or neglected appearance  

- **1 (Average)**  
  - Basic upkeep  
  - Some uneven growth or minor clutter  

- **2 (Good)**  
  - Well-maintained lawn or landscaping  
  - Clean, organized outdoor space  

---

### 3. Driveway / Walkway Condition (0–2)
Evaluates accessibility and surface condition.

- **0 (Poor)**  
  - Cracked, broken, or heavily damaged surfaces  
  - Obstructed or unsafe pathways  

- **1 (Average)**  
  - Usable but visibly worn  
  - Minor cracks or aging  

- **2 (Good)**  
  - Clean, intact surfaces  
  - No visible damage  

---

### 4. Immediate Street Condition (0–2)
Evaluates the visible road or street area directly adjacent to the property.

- **0 (Poor)**  
  - Significant cracks, potholes, debris  
  - Visibly degraded infrastructure  

- **1 (Average)**  
  - Functional but worn  
  - Minor imperfections  

- **2 (Good)**  
  - Clean, well-maintained road  
  - No visible damage  

---

## Final Score Calculation

Total Score =  
House + Yard + Driveway + Street  

Range: **0 – 8**

---

## Class Assignment

- **0 – 3 → Poor Condition**  
- **4 – 6 → Average Condition**  
- **7 – 8 → Well-Maintained Condition**

---

## Notes

- All scores must be assigned based solely on visible features in the image.
- Do not use external information (e.g., location, price, assumptions).
- If a category is not visible, assign a neutral score of **1 (Average)**.
- Consistency is prioritized over perfection.

---