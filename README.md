# Credit Risk Modelling for Loan Default

This project builds a classification model to predict the probability of mortgage loan default using applicant financial and demographic data. The model is deployed using a Flask web interface for user-friendly interaction.

## Project Overview

- **Objective**: Predict loan default likelihood to enable rejection of high-risk profiles.
- **Model Used**: Logistic Regression with class balancing.
- **Features**: Income, Loan Amount, Term Length, Occupation, Marital Status, SCHUFA Score, Number of Applications, Installment-to-Income Ratio.
- **Imbalance Handling**: Used `class_weight='balanced'` to address class imbalance.
- **Performance Metrics**:
  - ROC-AUC: **0.92**
  - Gini (OOT): **0.83**
  - Top 3 deciles captured **85%+** of actual defaulters.
  - Lift in top 3 deciles: **3x** (Train & OOT)

## Techniques Used

- Feature engineering and imputation
- Outlier detection via quantiles
- One-Hot Encoding for categorical variables
- Used SHAP summary plot to extract important features.
- Model evaluation via:
  - Confusion Matrix
  - ROC-AUC Curve
  - Gini Coefficient
  - Decile-based Rank Ordering, Capture rate & lift analysis

## Web Deployment (Flask)

A simple web app allows users to input applicant data and receive a real-time prediction of default risk.

### UI Features:
- HTML form with input fields (text and dropdowns)
- Flask backend to process and predict
- Prediction displayed on the same page


## Containerization (Docker)

The application has been dockerized completely and the Docker image has been pushed to DockerHub. Now anyone can take a pull of the image and run it on their system.
