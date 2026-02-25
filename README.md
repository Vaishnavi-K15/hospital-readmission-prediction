# Hospital Readmission Prediction

## Project Overview
Machine learning model to predict 30-day hospital readmissions, enabling proactive patient care and reducing readmission costs.

## Business Problem
- Hospital readmissions cost U.S. healthcare system $41 billion annually
- Medicare imposes financial penalties on hospitals with high readmission rates
- Need to identify high-risk patients for targeted interventions

## Dataset
- **Source:** UCI Machine Learning Repository - Diabetes 130 US Hospitals (1999-2008)
- **Records:** 101,766 hospital admissions
- **Features:** 50+ variables including patient demographics, diagnoses, medications, and procedures
- **Target:** 30-day readmission (11.16% positive class)

## Technical Approach

### 1. Data Cleaning & Preprocessing
- Handled missing values (weight: 96.86%, medical_specialty: 49.08%)
- Replaced '?' placeholders with appropriate values
- Simplified ICD-9 diagnosis codes into 11 disease categories
- Created binary target variable (readmitted within 30 days)

### 2. Feature Engineering
Created 11 new predictive features:
- `total_medications`: Count of prescribed diabetes medications
- `med_changed`: Binary flag for medication changes during visit
- `age_group`: Ordinal encoding of age ranges
- `num_diagnoses`: Count of diagnosis codes
- `total_prior_visits`: Sum of outpatient, emergency, and inpatient visits
- `utilization_score`: Composite metric of healthcare service usage
- Prior visit flags (outpatient, emergency, inpatient)

### 3. Model Development
**Models Tested:**
- Logistic Regression (baseline)
- Random Forest
- **XGBoost (selected model)**

**Handling Class Imbalance:**
- Applied class weights (8:1 ratio)
- Optimized decision threshold from 0.5 to 0.418

**Final Model Performance:**
- **Recall:** 70.0% (catching 7 out of 10 readmitted patients)
- **Precision:** 15.9%
- **F1-Score:** 0.260
- **ROC-AUC:** 0.676

### 4. Model Optimization
- Threshold tuning to maximize recall (healthcare priority: catch high-risk patients)
- Improved recall from 53.1% → 70.0% (+16.9 percentage points)
- Trade-off: Lower precision acceptable in healthcare context

## Key Findings

### Top 5 Risk Factors:
1. **Number of prior inpatient visits** (16.9% importance)
2. **Discharge disposition** (8.2% importance)
3. **Diabetes medication status** (7.1% importance)
4. **Total prior hospital visits** (6.4% importance)
5. **Glyburide-metformin usage** (5.9% importance)

### Business Impact
- **Patients Identified:** 7,949 high-risk patients annually
- **Prevented Readmissions:** 3,179 (assuming 40% intervention success rate)
- **Annual Cost Savings:** $43.71M
- **ROI:** 1100% return on intervention investment

## Project Structure
```
Hospital_Readmission_Project/
│
├── data/
│   ├── hospital_data_cleaned.csv      # Cleaned data
│   └── hospital_data_features.csv     # Feature-engineered data
│
├── notebooks/
│   └── Hospital_Readmission_Analysis.ipynb
│
├── models/
│   ├── final_readmission_model.pkl    # Trained XGBoost model
│   └── feature_scaler.pkl             # StandardScaler for features
│
├── dashboard/
│   ├── dashboard_data.csv             # Data for Tableau/Power BI
│   └── top_features.csv               # Feature importance
│
├── results/
│   ├── project_summary.csv            # Key metrics summary
│   └── feature_importance.csv         # Full feature importance ranking
│
└── README.md
```

## Tools & Technologies
- **Python 3.12:** pandas, numpy, scikit-learn, XGBoost, imbalanced-learn
- **Visualization:** Matplotlib, Seaborn, Tableau/Power BI
- **ML Techniques:** Feature engineering, class imbalance handling, threshold optimization
- **Version Control:** Git/GitHub

## Key Learnings
1. **Domain Knowledge Matters:** Understanding healthcare context shaped model optimization toward recall over precision
2. **Feature Engineering Impact:** Created features (prior visits, medication changes) were among top predictors
3. **Threshold Tuning:** Adjusting decision threshold from 0.5 to 0.418 significantly improved recall (53% → 70%)
4. **Class Imbalance:** Using class weights outperformed SMOTE for this dataset
5. **Business Translation:** Connected technical metrics to dollar impact (1100% ROI)
