[README.md](https://github.com/user-attachments/files/29850553/README.md)
# Credit Card Fraud Detection — Applied Machine Learning for a FinTech Startup

Binary classification project detecting fraudulent credit card transactions in a highly imbalanced, real-world-style dataset (fraud rate: **0.17%**). Built as part of a Higher Diploma in Data Analytics, this project simulates the kind of risk/fraud problem faced by fintech and payments companies.

## Problem

Credit card fraud detection is a classic extreme class-imbalance problem: of 284,807 transactions, only 492 (0.17%) are fraudulent. A model that simply predicts "not fraud" every time would be 99.8% accurate and completely useless. The goal here was to build a model that reliably catches fraud while keeping false positives manageable — and to be able to explain *why* it flags a transaction, which matters in a regulated financial environment.

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 European card transactions over two days, with 28 anonymized PCA features (`V1`–`V28`), plus `Time` and `Amount`. Not included in this repo due to size — download it from Kaggle and place `creditcard.csv` in the project root to reproduce.

## Approach

1. **EDA** — checked class distribution, missing values, and transaction amount patterns by class.
2. **Feature engineering** — derived `hour_of_day` and `is_night` from `Time`, and `log_amount` from `Amount` to reduce skew.
3. **Modelling** — trained two models on a stratified 80/20 train/test split:
   - **Logistic Regression** (baseline) — scaled features, class-balanced, chosen for interpretability and as a sanity-check linear benchmark.
   - **Random Forest** (primary model) — chosen because the dataset's PCA-anonymized features involve non-linear interactions that a linear model struggles to capture; Random Forest handles this natively, is robust to noise, and supports feature-importance-based interpretability.
4. **Evaluation** — since accuracy is meaningless under 0.17% class imbalance, the model was evaluated on **recall, precision, F1, ROC-AUC and PR-AUC** for the fraud class specifically.
5. **Explainability (XAI)** — used **SHAP** to explain both global feature importance and individual predictions, including a waterfall plot for a specific fraud case.

## Results

| Model | ROC-AUC | PR-AUC | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.972 | 0.723 | 0.058 | 0.918 | 0.109 |
| **Random Forest (primary)** | 0.957 | **0.871** | **0.961** | 0.745 | **0.839** |

The Random Forest model catches **74.5% of fraudulent transactions** while keeping precision at **96%** — meaning very few legitimate transactions are wrongly flagged, which is critical for customer experience in a production fraud system. PR-AUC (the more informative metric under this level of imbalance) improved from 0.723 to 0.871 over the linear baseline.

### Explainability
SHAP was used to make the Random Forest's decisions transparent — both which features matter most globally, and why any single transaction was flagged. In one example, a transaction was scored at **0.95 fraud probability**, driven primarily by features `V14`, `V10`, `V12` and `V17`. This kind of instance-level explanation supports manual review workflows, regulatory auditability, and threshold calibration in a real fraud system.

## Stack

Python · pandas · NumPy · scikit-learn · SHAP · matplotlib

## Repo structure

```
├── fraud_detection_fintech.ipynb   # Full analysis: EDA → feature engineering → modelling → evaluation → SHAP
├── README.md
└── requirements.txt
```

## Reproducing this

```bash
pip install -r requirements.txt
# download creditcard.csv from Kaggle into this folder
jupyter notebook fraud_detection_fintech.ipynb
```
