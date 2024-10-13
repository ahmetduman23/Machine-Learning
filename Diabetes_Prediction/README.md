# Feature Engineering for Diabetes Prediction

This project focuses on predicting diabetes using the Pima Indian Diabetes dataset from the National Institute of Diabetes and Digestive and Kidney Diseases. The goal is to predict whether a patient has diabetes based on various diagnostic measurements. All patients in this dataset are female, at least 21 years old, and of Pima Indian heritage.

## Dataset Overview

- **Source:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Objective:** To predict the likelihood of diabetes based on diagnostic measurements.
  
### Features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (2 hours after a glucose tolerance test)
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history
- **Age**: Age in years

### Target:
- **Outcome**: Whether or not the patient has diabetes (1 = Yes, 0 = No)

## Steps in the Project

### 1. Data Analysis
- **Overview:** General inspection of the dataset, identifying missing values, and understanding the structure.
- **Categorical Variable Analysis:** Examination of categorical data distributions.
- **Numerical Variable Analysis:** Detailed analysis of the distribution and relationships of numerical features.
- **Target Variable Analysis:** Investigating the balance between diabetic and non-diabetic patients.
- **Correlation Matrix:** Displaying feature correlations to understand interdependencies.

### 2. Feature Engineering
- **Missing Value Analysis:** Identifying and handling missing data appropriately.
- **Feature Extraction:** Creation of new features based on domain knowledge or data manipulation.
- **Encoding:** Converting categorical variables into numerical representations.
- **Standardization:** Scaling features to standardize data for model training.
- **Model Setup:** Building and training the machine learning model.
- **Feature Importance:** Determining which features contribute most to model accuracy.

## Model Performance

| Metric      | Value  |
|-------------|--------|
| **Accuracy**| 0.79   |
| **Recall**  | 0.711  |
| **Precision**| 0.67  |
| **F1 Score**| 0.69   |
| **AUC**    | 0.77   |

The model's performance shows a balanced trade-off between precision, recall, and accuracy, making it a reliable predictor for diabetes diagnosis.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmetduman23/diabetes-prediction



Check out the full implementation on Kaggle: (https://www.kaggle.com/code/ahmetyasirduman/feature-engineering-for-diabetes-prediction)


Author

Ahmet Yasir Duman

GitHub: https://github.com/ahmetduman23?tab=repositories
Kaggle: https://www.kaggle.com/ahmetyasirduman

