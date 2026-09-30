# E-commerce Return Risk Classification

A machine learning prototype that predicts whether an e-commerce order is likely to be **returned**, using order details, customer history, and delivery information. Built as a capstone project for Learn Depth Academy — Track 1, Final Capstone, Problem 27.

---

## Problem Statement

Classify orders into low-risk and high-risk return categories based on measurable order and customer-history characteristics. The goal is an educational end-to-end ML pipeline — from data investigation through a deployable lightweight prediction app — using **only free and open-source tools**.

- **Domain:** E-commerce
- **ML Task:** Binary classification
- **Target:** `returned` (0 = not returned, 1 = returned)

---

## Dataset

- **Source:** Synthetic e-commerce order dataset (6,000 rows, 12 columns).
- **Features used:**
  - **Order details:** `order_value`, `item_count`, `discount_pct`, `delivery_time_days`
  - **Customer history:** `customer_prior_orders`, `customer_prior_returns`, `customer_return_rate`
  - **Categorical:** `category`, `payment_method`
- **Dropped columns:** `order_id`, `customer_id` — identifiers with no predictive value
- **Missing values:** `delivery_time_days` (~1%) — median imputation
- **Class balance:** ~66% not returned / ~34% returned

---

## Methodology

1. **Exploratory Data Analysis** — data quality, missing values, distributions, bivariate analysis, correlation, outliers, leakage check.
2. **Preprocessing** — median imputation, StandardScaler on numeric features, One-Hot Encoding on categorical features. All encapsulated in a `ColumnTransformer` pipeline, fit on training data only.
3. **Baseline Models** — Logistic Regression, KNN, Decision Tree.
4. **Improvement** — KNN with RandomOverSampler, Decision Tree hyperparameter tuning via GridSearchCV, Logistic Regression decision-threshold tuning.
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, 5-fold stratified cross-validation.

---

## Results Summary

| Model | F1 (Returned) | ROC-AUC |
|---|---|---|
| LR (baseline) | 0.580 | 0.739 |
| KNN + oversampling | 0.530 | 0.659 |
| DT (tuned) | 0.553 | 0.686 |
| **LR @ threshold 0.45** | **0.583** | **0.739** |

**Final model:** Logistic Regression with a decision threshold of **0.45**, chosen for the highest F1-score on the minority ("Returned") class, the highest ROC-AUC, and full coefficient-based interpretability.

---

## Project Structure

```
Capstone project/
├── app.py                          Streamlit application
├── final_return_risk_model.pkl     Trained model + tuned threshold
├── requirements.txt                Python dependencies
├── README.md                       This file
├── ecommerce_return_risk.xlsx      Source dataset
└── outputs/                        Saved charts and screenshots
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Using the App

1. Fill in the order details — order value, item count, discount percentage, delivery time.
2. Fill in customer history — prior orders, prior returns, historical return rate.
3. Select product category and payment method.
4. Click **Predict Return Risk**.
5. The app displays:
   - **Class prediction** — Low or High Return Risk
   - **Probability of return** — a percentage
   - **Top contributing factors** — with direction (🔺 pushes toward return, 🔻 pushes away)

Input validation prevents inconsistent values — for example, prior returns exceeding prior orders.

---

## Technical Notes

- **Reproducibility:** All preprocessing steps are encapsulated inside the saved scikit-learn pipeline, so the app applies identical transformations to what was used during training.
- **Class imbalance:** Handled via `class_weight='balanced'` (Logistic Regression, Decision Tree) and RandomOverSampler (KNN).
- **Threshold tuning:** The default 0.50 decision threshold was replaced with 0.45 after evaluating the precision/recall trade-off — improving recall on returns from 0.65 to 0.73.
- **No data leakage:** Verified that `customer_return_rate` is strictly historical (matches `prior_returns / prior_orders`), and identifier columns were removed before training.

---

## Limitations

- Educational prototype — **not** production-ready.
- Trained on a single synthetic dataset; may not generalize to real-world markets.
- Best F1-score on the minority class is ~0.58 — return prediction is inherently noisy with these features.
- No time-based validation; the model assumes stationarity of return patterns.

---

## Future Scope

- Integrate product-level return history and review sentiment signals.
- Add gradient boosting models (XGBoost, LightGBM) for comparison.
- Deploy with drift monitoring and fairness checks across customer segments.
- Explore cost-sensitive learning — weighting false negatives more heavily than false positives.

---

## References

- Scikit-learn documentation — https://scikit-learn.org
- Streamlit documentation — https://docs.streamlit.io
- Imbalanced-learn documentation — https://imbalanced-learn.org

---

## Author

**Abdullah**
Learn Depth Academy — Track 1 Final Capstone, Problem 27
