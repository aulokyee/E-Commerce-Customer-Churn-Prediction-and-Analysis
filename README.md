# E-Commerce Churn Prediction Project

## Project Proposal

### Overview
This project aims to predict customer churn in an e-commerce platform using machine learning and visualize key insights to inform retention strategies. Churn, defined as customers ceasing to engage with the platform, affects ~16.84% of customers in the dataset. By identifying churn drivers (e.g., low satisfaction, short tenure, complaints), the project supports targeted interventions to improve retention.

### Objectives
1. **Predict Churn**: Develop machine learning models to identify customers likely to churn.
2. **Identify Drivers**: Analyze features (e.g., `SatisfactionScore`, `Tenure`, `Complain`) to understand churn causes.
3. **Visualize Insights**: Create Tableau dashboards using raw data to present churn patterns and actionable recommendations.
4. **Evaluate Models**: Compare Random Forest, XGBoost, and Logistic Regression to select the best performer.

### Dataset
- **File**: `cleaned_data_no_encoding.csv`
- **Size**: 5,630 customers, 20 features
- **Key Features**:
  - `Churn`: Binary target (1 = churned, 0 = unchurned, ~16.84% churn rate).
  - `SatisfactionScore`: Customer satisfaction (1–5, ~30% churn for scores 1–2).
  - `Tenure`: Months since first purchase (~40% churn for <6 months).
  - `Complain`: Binary (1 = complaint filed, ~30–40% churn for complainers).
  - Others: `OrderCount`, `CashbackAmount`, `MaritalStatus`, `CityTier`, `PreferredOrderCat`, etc.


## Steps Taken to Tackle the Problem

### 1. Data Exploration and Preprocessing
- **Exploration**:
  - Used Python (pandas) to analyze the dataset, confirming 5,630 rows and ~948 churned customers.
  - Identified key churn drivers: low `Cashback Amount` , short `Tenure` (<6 months), high `Complain`.
  - Verified no missing values; features were numeric or categorical.
- **Preprocessing**:
  - Split data into train (80%) and test (20%) sets, stratified by `Churn` to maintain ~16.84% churn rate.
  - Selected top 10 features (e.g., `SatisfactionScore`, `Tenure`, `Complain`) using correlation or feature importance.
  - No additional encoding required (dataset pre-cleaned).

### 2. Model Training and Evaluation
- **Models Trained**:
  - **Random Forest**: Ensemble model with top 10 features.
  - **XGBoost**: Tuned model (e.g., `max_depth`, `learning_rate`, `scale_pos_weight`) for imbalanced data.
  - **Logistic Regression**: Tuned linear model as a baseline.
- **Evaluation Metrics**:
  - **F1 Score**: Balances precision and recall, critical for imbalanced data (~16.84% positive class).
  - **ROC AUC**: Measures ability to distinguish churned vs. unchurned customers.
- **Results**:
  - **Random Forest**:
    - Train F1: ~0.829, Test F1: ~0.610
    - Train ROC AUC: ~0.916, Test ROC AUC: ~0.870
  - **XGBoost (Tuned)**:
    - Train F1: ~0.899, Test F1: ~0.659
    - Train ROC AUC: 0.964, Test ROC AUC: ~0.900
  - **Logistic Regression**:
    - Train F1: 0.815, Test F1: 0.540
    - Train ROC AUC: ~0.890, Test ROC AUC: ~0.852
- **Conclusion**: XGBoost outperformed others (Test F1: ~0.659, Test ROC AUC: ~0.900), offering the best balance for minority class prediction. Saved as `tuned_xgb_top10_final_retuned.joblib`.

### 3. Model Deployment
- Generated predictions using XGBoost for analysis:
  - Output: `churn_predictions.csv` with `CustomerID`, `Churn` (actual), `Churn_Probability`.
  - Note: Predictions were not used in Tableau dashboards, which rely solely on raw data.

### 4. Visualization with Tableau
- **Objective**: Visualize churn patterns and retention opportunities using `cleaned_data_no_encoding.csv`, excluding machine learning predictions.
- **Dashboards Created**:
  1. **Churn Analysis Dashboard** (`EcommerceDashboard1.twb`):
     - **Churn Rate KPI**: Displays ~16.84% churn rate.
     - **Churn by Satisfaction and City Tier**: Bar chart showing ~30% churn for `SatisfactionScore`=1–2 across `CityTier`.
     - **Churn by Tenure**: Histogram showing ~40% churn for `Tenure` <6 months.
  2. **Retention Opportunities Dashboard** (`Retention_Opportunities.twb`):
     - **Churn by Complaint Status**: Bar chart showing ~30–40% churn for `Complain`=1.
     - **Churn by Order Count and Cashback**: Bar plot linking low `CashbackAmount` to churn.
- **Interactivity**: Added filters (e.g., `Gender`, `CityTier`, `PreferredOrderCat`) and actions for stakeholder exploration.


### 5. Insights
- **Churn Drivers**: Low `Cashback Amount`, short `Tenure`, and high `Complain` are key predictors, as shown in Tableau dashboards.
- **Retention Strategies**: Enhance customer support for low satisfaction, improve onboarding for new customers, resolve complaints promptly, and offer cashback to low-engagement customers.

## Instructions to Run the Project

### Prerequisites
- **Python 3.8+**: For model training and predictions.
- **Tableau Desktop or Public**: For dashboard visualization.
- **Dependencies**:
  ```bash
  pip install pandas scikit-learn xgboost joblib matplotlib