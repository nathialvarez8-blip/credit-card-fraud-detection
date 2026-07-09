#  Credit Card Fraud Detection  
### Applied Machine Learning for a FinTech Use Case

##  Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques.  
The goal is to build, evaluate, and explain predictive models capable of identifying rare fraud cases in a highly imbalanced dataset, simulating a real-world FinTech scenario.



##  Objectives
- Analyze and understand patterns in credit card transaction data  
- Handle class imbalance effectively  
- Train and compare baseline and advanced classification models  
- Evaluate model performance using appropriate metrics  
- Apply explainable AI (XAI) techniques to interpret model decisions  



##  Models Used
- **Logistic Regression** (Baseline model)  
- **Random Forest Classifier** (Primary model)

Both models were trained using class-weighting strategies to address class imbalance.



## Tools & Technologies
- Python  
- Pandas, NumPy  
- Matplotlib  
- Scikit-learn  
- SHAP (Explainable AI)  
- Jupyter Notebook  



##  Dataset
The dataset contains anonymized credit card transactions, where:
- `Class = 0` → Legitimate transaction  
- `Class = 1` → Fraudulent transaction  

Due to confidentiality, most features are PCA-transformed variables (V1–V28), along with:
- `Time`  
- `Amount`



##  Exploratory Data Analysis (EDA)
- Analysis of severe class imbalance  
- Visualization of fraud vs legitimate transactions  
- Summary statistics of transaction amounts by class  



##  Feature Engineering
Additional features were created to enhance model performance:
- `hour_of_day`
- `is_night`
- `log_amount`



## Evaluation Metrics
Models were evaluated using:
- Precision, Recall, F1-score  
- Confusion Matrix  
- ROC-AUC  
- Precision–Recall Curve  

These metrics are particularly important for fraud detection, where false negatives are costly.


## Model Comparison
Logistic Regression was used as a baseline due to its simplicity and interpretability.  
Random Forest was selected as the primary model due to its ability to capture non-linear relationships and improve fraud detection performance.



## Explainable AI (XAI)
SHAP (SHapley Additive exPlanations) was applied to the Random Forest model to:
- Understand global feature importance  
- Explain individual fraud predictions  
- Increase model transparency and trustworthiness  

Both global and local explanations were generated.



## 🚀 How to Run
Open the notebook using Jupyter Notebook or Google Colab.

