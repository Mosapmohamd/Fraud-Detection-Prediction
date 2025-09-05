# AI-Powered Fraud Detection System

## Overview
This project is a **Graduation Project** that focuses on building an AI-based system for detecting fraudulent financial transactions.  
The system applies different sampling strategies and machine learning models to handle **class imbalance** and achieve high accuracy in fraud detection.

---

## Problem
Financial fraud is a major challenge that causes **billions of dollars in losses** for banks and businesses every year.  
Detecting fraud is difficult due to the **rarity of fraudulent transactions** compared to normal ones.

---

## Objectives
- Build a machine learning pipeline to predict fraudulent transactions.  
- Compare different resampling strategies (Original, Under-sampling, Over-sampling, SMOTE).  
- Evaluate multiple models to identify the best-performing one.  
- Develop a practical dashboard to demonstrate fraud detection in action.

---

## Dataset
- Source: Financial transactions dataset.  
- Size: Millions of rows with multiple features.  
- Key features: `amount`, `oldbalanceOrg`, `newbalanceDest`, `type`, etc.  
- Target: `isFraud` (1 = Fraud, 0 = Not Fraud).

---

## Methodology
1. **Data Preprocessing**  
   - Dropped irrelevant columns (`nameOrig`, `nameDest`).  
   - Scaled numeric features and one-hot encoded categorical ones.  

2. **Handling Imbalance**  
   - Tested multiple strategies:  
     - Original data  
     - Random Under-sampling  
     - Random Over-sampling  
     - SMOTE  

3. **Model Training**  
   - Models used:  
     - Logistic Regression  
     - Random Forest  
     - LightGBM  
     - XGBoost  
   - Cross-validation: **Stratified K-Fold (5 folds)**.  

4. **Evaluation Metrics**  
   - Precision  
   - Recall  
   - F1-Score  
   - AUC  

---

## Results
- Models performed differently across sampling strategies.  
- Example outcome: **SMOTE + LightGBM achieved the highest F1 Score and AUC**.  
- Results were summarized in comparative tables and plots.

---

## Best Model
- **LightGBM with SMOTE** was selected as the best model.  
- Achieved strong balance between **Recall and Precision**.  

---

## Practical Application
The model can be integrated into **banking systems** to detect fraud in real time.  
A Streamlit-based dashboard demonstrates:  
- User inputs transaction details.  
- Model predicts fraud likelihood.  
- Alert generated if suspicious.  

---

## Challenges
- Highly imbalanced dataset.  
- Large data size and training time.  
- Choosing the right sampling strategy.  

---

## Future Work
- Deploy the system on the cloud.  
- Integrate with live transaction systems.  
- Explore deep learning approaches.  



---
