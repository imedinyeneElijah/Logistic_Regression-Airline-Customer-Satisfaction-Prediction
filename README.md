# Logistic_Regression-Airline-Customer-Satisfaction-Prediction
The project demonstrates how machine learning can help airlines identify key drivers of customer satisfaction and support data-driven business decisions aimed at improving customer experience and retention.


## Project Overview

This project builds a binary Logistic Regression classifier to predict whether an Invistico Airline passenger is **satisfied** or **dissatisfied**, using 22 features from post-flight surveys. The goal is to identify which service dimensions most strongly influence satisfaction and translate those findings into data-backed business recommendations.

---

## Dataset Description

**File:** `Invistico_Airline.csv`  
**Rows:** 129,880 passenger survey records  
**Target:** `satisfaction` → binary (satisfied / dissatisfied)

| Column | Type | Description |
|--------|------|-------------|
| `satisfaction` | Categorical | **Target**: satisfied / dissatisfied |
| `Customer Type` | Categorical | Loyal / disloyal customer |
| `Age` | Numeric | Passenger age |
| `Type of Travel` | Categorical | Business / Personal |
| `Class` | Categorical | Business / Eco Plus / Eco |
| `Flight Distance` | Numeric | Flight distance in miles |
| `Seat comfort` | Integer (0–5) | Rating |
| `Inflight wifi service` | Integer (0–5) | Rating |
| `Inflight entertainment` | Integer (0–5) | Rating |
| `Online boarding` | Integer (0–5) | Rating |
| `Food and drink` | Integer (0–5) | Rating |
| `Gate location` | Integer (0–5) | Rating |
| `Online support` | Integer (0–5) | Rating |
| `Ease of Online booking` | Integer (0–5) | Rating |
| `On-board service` | Integer (0–5) | Rating |
| `Leg room service` | Integer (0–5) | Rating |
| `Baggage handling` | Integer (0–5) | Rating |
| `Checkin service` | Integer (0–5) | Rating |
| `Cleanliness` | Integer (0–5) | Rating |
| `Departure/Arrival time convenient` | Integer (0–5) | Rating |
| `Departure Delay in Minutes` | Numeric | Departure delay |
| `Arrival Delay in Minutes` | Numeric | Arrival delay (393 missing → median imputed) |

---

## Modeling Approach

1. **Preprocessing**
   - Median imputation for `Arrival Delay in Minutes` (0.30% missing)
   - Ordinal target encoding: satisfied=1, dissatisfied=0
   - One-hot encoding for `Customer Type`, `Type of Travel`, `Class` (drop_first=True)
   - StandardScaler fitted on training data only (no data leakage)

2. **Train/Test Split:** 80/20 stratified split (103,904 train / 25,976 test)

3. **Model:** Scikit-learn `LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000)`

---

## Final Performance Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 82.89% | Overall correct predictions |
| **Precision** | 0.8441 | 84.4% of passengers flagged Satisfied truly are |
| **Recall** | 0.8430 | 84.3% of truly Satisfied passengers correctly identified |
| **F1 Score** | 0.8436 | Balanced precision-recall performance |
| **ROC-AUC** | 0.9035 | Strong discrimination ability |

### Confusion Matrix (Test Set — 25,976 passengers)

|  | Predicted Dissatisfied | Predicted Satisfied |
|--|------------------------|---------------------|
| **Actual Dissatisfied** | 9,546 ✅ | 2,213 ❌ |
| **Actual Satisfied** | 2,232 ❌ | 11,985 ✅ |

---

## Key Findings — Top Satisfaction Drivers

| Feature | Odds Ratio | Effect |
|---------|-----------|--------|
| Inflight entertainment | 2.65 | +164.6% odds increase per SD |
| Seat comfort | 1.48 | +48.1% odds increase per SD |
| On-board service | 1.47 | +47.3% odds increase per SD |
| Checkin service | 1.43 | +42.7% odds increase per SD |
| **Customer Type (disloyal)** | 0.48 | -51.6% odds decrease |
| **Eco class** | 0.70 | -29.9% odds decrease |
| **Personal Travel** | 0.71 | -29.5% odds decrease |

---

## Business Recommendations

1. **Inflight Entertainment** is the single strongest predictor — refresh cabin entertainment systems on low-satisfaction routes first.
2. **Seat comfort & leg room** are physical touchpoints that drive significant lift — prioritise retrofit on high-traffic routes.
3. **Online boarding & digital experience** are high-coefficient, low-cost improvements — redesign the mobile boarding flow.
4. **Disloyal customers** have a 51.6% lower odds of satisfaction — target this segment with loyalty onboarding campaigns.
5. **Arrival delay** hurts satisfaction, but its coefficient is smaller than service quality features — invest in service recovery quality, not just operational delays.

---

## Environment Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**Python version:** 3.9+

---

## Project Structure

```
.
├── logistic_regression_analysis.ipynb   # Main notebook (all cells executed)
├── Invistico_Airline.csv                # Dataset
└── README.md                            # This file
```

## Running the Notebook

```bash
jupyter notebook logistic_regression_analysis.ipynb
```

Re-run with `Kernel > Restart & Run All`.

---

## Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup | Imports and display settings |
| 2. Load & Inspect | Dataset shape, dtypes, missing values |
| 2.1 EDA | Class balance, satisfaction by category, service ratings, age/distance |
| 3. Preprocessing | Imputation, target encoding, one-hot encoding, scaling, train/test split |
| 4. Model | Logistic Regression training |
| 5. Evaluation | Confusion matrix, Precision, Recall, F1, ROC-AUC, ROC curve |
| 6. Coefficients | Odds ratios, coefficient plot, driver interpretation |
| 7. Performance Summary | Full metrics table |
| 8. Business Recommendations | Prioritised actions + model limitations + threshold analysis |


