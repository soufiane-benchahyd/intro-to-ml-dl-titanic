# Introduction to Machine Learning – Titanic Survival Prediction

## Project Overview
This project is an introductory hands-on application of **Artificial Intelligence and Machine Learning**.
The objective is to predict whether a passenger survived the Titanic disaster based on personal and socio-economic features.

The project covers the **full machine learning pipeline**, from data preprocessing and exploratory analysis
to model comparison and deployment in a web application.

This work was developed as part of the *Artificial Intelligence* module.

---

## Objectives
- Build a complete machine learning pipeline
- Compare multiple classification algorithms
- Evaluate models using accuracy and ROC-AUC
- Deploy the best-performing model using Streamlit

---

## Dataset
- **Source**: Titanic dataset (Kaggle)
- **Samples**: 891 passengers
- **Target variable**: `Survived` (0 = No, 1 = Yes)
- **Features used**:
  - Pclass
  - Sex
  - Age
  - SibSp
  - Parch
  - Fare
  - Embarked
  - Engineered features (FamilySize, IsAlone, HasCabin)

---

## Data Preprocessing
- Missing ages replaced by the median
- Missing embarkation ports filled with the most frequent value
- Categorical variables encoded numerically
- Feature engineering to improve predictive power
- Stratified train/test split (70% / 30%)

---

## Models Implemented
The following classification models were trained and compared:

- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

---

## Model Performance

| Model               | Accuracy | ROC-AUC |
|--------------------|----------|---------|
| Logistic Regression| 79.48%   | **0.849** |
| Random Forest      | **81.34%** | 0.846 |
| KNN                | 69.40%   | 0.701 |
| SVM                | 63.81%   | 0.696 |

### Selected Model: Logistic Regression
The logistic regression model was selected due to:
- Best ROC-AUC score
- High interpretability
- Fast training and inference
- Good balance between simplicity and performance

---

## Deployment
The selected model was deployed using **Streamlit** as an interactive web application.

### Features:
- User-friendly form interface
- Real-time prediction
- Probability confidence score
- Visual feedback

### Run the application locally:
```bash
pip install -r requirements.txt
streamlit run app.py
