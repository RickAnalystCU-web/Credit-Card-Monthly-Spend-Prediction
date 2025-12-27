# Credit Card Monthly Spend Prediction (Business Data Analytics)

A lightweight customer-level prediction project in a **business context**: estimate each customer’s **monthly credit card spend** to support **budgeting accuracy** and **customer targeting / planning** discussions.

This repo contains a simple end-to-end workflow from feature prep to model training and scoring.

---

## Business Context (Why this matters)
Companies that manage consumer finance products often need a reliable estimate of customer spend to:
- improve budgeting / forecasting
- support customer segmentation and targeting
- enable scenario planning (e.g., expected monthly spend under different conditions)

This project frames the task as a **customer-level regression problem**: predict `monthly_spend` using demographic, credit, and behavioral attributes.

---

## What the Project Does
- Loads training data (`analysis_data.csv`) and scoring data (`scoring_data.csv`)
- Builds an analysis-ready feature matrix:
  - drops identifier/label columns (e.g., `customer_id`, `monthly_spend`)
  - applies **one-hot encoding** to categorical features
  - applies **PolynomialFeatures** on numeric features to capture non-linear signals
- Trains a **Ridge Regression** model (regularized linear model)
- Generates a prediction file: `submission_ridge_poly_final.csv`

---

## Modeling Approach (Quick Summary)
- **Model**: Ridge Regression (`sklearn.linear_model.Ridge`)
- **Feature engineering**:
  - one-hot encoding for categorical columns
  - polynomial expansion for numeric columns (`sklearn.preprocessing.PolynomialFeatures`)
- **Reasoning**: a regularized linear model + lightweight feature engineering gives a stable baseline that balances:
  - predictive accuracy
  - robustness / stability
  - interpretability

> Result highlight (from project notes): RMSE ≈ **247.67**, within **0.42** of the best observed benchmark (**247.25**).

---

## Files
- `linear+poly_final.ipynb` — main notebook (training + scoring + CSV export)
- `analysis_data.csv` — training dataset (includes target `monthly_spend`)
- `scoring_data.csv` — dataset to score (no target)
- `submission_ridge_poly_final.csv` — output predictions (generated)

---

## How to Run

### 1) (Optional) Create & activate a virtual environment
```bash
python -m venv venv
````

Activate it:

* macOS / Linux

  ```bash
  source venv/bin/activate
  ```
* Windows (PowerShell / CMD)

  ```bash
  venv\Scripts\activate
  ```

### 2) Install dependencies (based on imports in the notebook)

This repo does **not** include a `requirements.txt`.
Install packages based on `import ...` statements used in the notebook.

Typical setup:

```bash
pip install pandas numpy scikit-learn jupyter
```

If you run into `ModuleNotFoundError`, install the missing package:

```bash
pip install <missing-package-name>
```

### 3) Run the notebook

```bash
jupyter notebook
```

Open `linear+poly_final.ipynb` and run all cells.

### 4) Output

After execution, the notebook writes:

* `submission_ridge_poly_final.csv`

Format:

* `customer_id`
* `monthly_spend` (predicted)

---

## Notes

* Column names / feature lists depend on your dataset schema; the notebook handles preprocessing based on defined categorical & numeric columns.
* This is a compact business-prediction baseline project meant to be readable and easy to extend (e.g., try different alphas, add feature selection, or compare with tree-based models).
