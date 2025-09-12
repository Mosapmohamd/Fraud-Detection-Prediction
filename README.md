# 💻 AI-Powered Fraud Detection System  

An AI-based system for detecting fraudulent financial transactions  
Developed as a Graduation Project during the **NTI & ITIDA Training Program**

---

## 📌 Overview  
Financial fraud causes billions in losses every year  
Detecting fraudulent transactions is difficult due to their rarity compared to legitimate ones  

This project applies advanced **machine learning models** and **sampling strategies** to handle class imbalance and achieve **high fraud detection accuracy**

---

## 🎯 Objectives  
- Build an end-to-end ML pipeline to predict fraudulent transactions  
- Apply and compare different resampling strategies  
- Evaluate multiple ML models using standard metrics  
- Deploy a practical dashboard to demonstrate fraud detection  

---

## 📂 Dataset  
- **Source**: Financial transactions dataset  
- **Size**: Millions of rows  
- **Key Features**:  
  - `amount`  
  - `oldbalanceOrg`  
  - `newbalanceDest`  
  - `type`  
- **Target**: `isFraud` (1 = Fraud, 0 = Not Fraud)  

---

## ⚙️ Methodology  

### Data Preprocessing  
- Removed irrelevant columns (`nameOrig`, `nameDest`)  
- Scaled numeric features  
- One-hot encoded categorical features  

### Handling Class Imbalance  
- Tested multiple strategies:  
  - Original data  
  - Random Under-sampling  
  - Random Over-sampling  
  - SMOTE  

### Model Training  
- Models:  
  - Logistic Regression  
  - Random Forest  
  - LightGBM  
  - XGBoost  
- Validation: Stratified K-Fold (5 folds)  

### Evaluation Metrics  
- Precision  
- Recall  
- F1-Score  
- AUC  

---

## 📊 Results  
- SMOTE + LightGBM achieved the **highest F1-Score and AUC**  
- Balanced precision and recall  
- Results summarized in comparative tables and plots  

---

## 🏆 Best Model  
- **LightGBM + SMOTE**  
- Selected for deployment due to strong performance on both precision and recall  

---

## 💻 Dashboard Application  
A **Streamlit-based dashboard** was built to demonstrate the model in action  
- User inputs transaction details  
- Model predicts fraud probability  
- System alerts if transaction is suspicious  

🔗 [Live App](https://fraud-detection-prediction.streamlit.app/)  

---

## 📁 Repository  
Explore the full project implementation, notebooks, and code  
🔗 [GitHub Repository](https://github.com/Mosapmohamd/Fraud-Detection-Prediction)  

---

## 👥 Team  
- **Mosap Mohamed**  
- **Jana Osame**  
- **Hana Salah**  

**Mentor:** Eng. Fatma  

---

## 🚀 Future Work  
- Integrate the model with real-time banking systems  
- Optimize model inference for production environments  
- Expand feature engineering for higher accuracy  
