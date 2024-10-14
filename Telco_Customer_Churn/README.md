# Feature Engineering for Telco Customer

This project aims to use customer churn data from a fictitious telecom company that provides home phone and internet services to 7043 customers in California to develop a machine learning model that can predict whether customers will abandon their services.

## Dataset Overview

- **Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Purpose:** Predicting the likelihood of customers leaving your services.

### Features:
- **CustomerId:** Customer ID
- **Gender:** Gender
- **SeniorCitizen:** Whether the customer is a senior citizen (1, 0)
- **Partner:** Whether the client has a partner (Yes, No)
- **Dependents:** Whether the client has dependents (Yes, No)
- **tenure:** Number of months the client has been with the company
- **PhoneService:** Whether the customer has phone service (Yes, No)
- **MultipleLines:** Whether the customer has more than one line (Yes, No, No phone service)
- **InternetService:** Customer's internet service provider (DSL, Fiber optic, No)
- **OnlineSecurity:** Whether the client has online security (Yes, No, No Internet service)
- **OnlineBackup:** Whether the customer has an online backup (Yes, No, No Internet service)
- **DeviceProtection:** Whether the customer has device protection (Yes, No, No Internet service)
- **TechSupport:** Whether the customer receives technical support (Yes, No, No Internet service)
- **StreamingTV:** Whether the customer has streaming TV (Yes, No, No Internet service)
- **StreamingMovies:** Whether the customer has streaming movies (Yes, No, No Internet service)
- Contract:** Client's contract duration (Month to month, One year, Two years)
- **PaperlessBilling:** Whether the customer has a paperless bill (Yes, No)
- **PaymentMethod:** Customer's payment method (Electronic check, Postal check, Bank transfer (automatic), Credit card (automatic))
- **MonthlyCharges:** Amount charged to the customer on a monthly basis
- **TotalCharges:** Total amount charged to the customer
- Whether the customer is using (Yes or No) - Customers who left in the last month or quarter

### Objective:
- **Outcome:** Predict whether customers will leave the company (1 = Yes, 0 = No)

## Steps in the Project

#### 1. Data Analysis
- Overview:** Overview of the data set, identification of missing values and understanding the structure.
- Categorical Variable Analysis:** Examination of categorical data distributions.
- Numerical Variable Analysis:** Detailed analysis of the distribution and relationships of numerical characteristics.
- Target Variable Analysis:** Investigation of the balance between abandoning and non-abandoning customers.
- Correlation Matrix:** Displaying feature correlations to understand interdependencies.

### 2. Feature Engineering
- Missing Value Analysis:** Appropriate identification and processing of missing data.
- Feature Extraction:** Creation of new features based on domain knowledge or data manipulation.
- Coding:** Transforming categorical variables into numerical representations.
- Standardization:** Scaling features to standardize data for model training.
- Model Setup:** Building and training a machine learning model.
- Feature Importance:** Determine which features contribute most to model accuracy.

## Model Performance

| Metric | Value |
|------------------|-------|
| **Accuracy** | 0.79 |
| **Recall** | 0.65 |
| **Sensitivity** | 0.50 |
| **F1 Score** | 0.56 |
| **AUC** | 0.74 |

The model's performance shows a balanced trade-off between precision, recall and accuracy, making it a reliable predictor for diagnosing whether customers will leave the company.

## How to Operate

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmetduman23/telco-customer-churn


Author

Ahmet Yasir Duman

GitHub: https://github.com/ahmetduman23?tab=repositories
Kaggle: https://www.kaggle.com/ahmetyasirduman
