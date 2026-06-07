# SafeFlow AI — Financial Crime Detection

> Transaction anomaly detection, money laundering identification, and scam behaviour analysis using machine learning.

---

## Overview

SafeFlow AI is an end-to-end machine learning pipeline for detecting financial crime in transaction data. It moves beyond simple classification to engineer **behavioural signals** — the same category of features used in real-world AML and fraud detection systems.

The pipeline detects three overlapping threat patterns:

| Threat Type | What It Looks Like |
|---|---|
| **Money Laundering** | Layering via currency conversion, cross-border structuring, fan-out/fan-in patterns |
| **Transaction Anomalies** | Amounts unusually large relative to an account's own history; late-night or weekend activity |
| **Scam Indicators** | Transactions to previously unseen recipients; sudden volume spikes from dormant accounts |

Built on the [SAML-D synthetic AML dataset](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml) (~9.5M transactions).

---

## Feature Engineering

### Temporal Features
| Feature | Description |
|---|---|
| `hour_` | Hour of transaction (0–23) |
| `day_` | Day of week (0=Mon, 6=Sun) |
| `month_` | Month of year |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_night` | 1 if between 10pm and 6am |

### Behavioural / Velocity Features
These are derived per `Sender_account` and `Receiver_account` across the full dataset. They capture *how this account behaves*, not just what a single transaction looks like.

| Feature | Description | Why It Matters |
|---|---|---|
| `sender_tx_count` | Total transactions from this sender | High counts can indicate smurfing or automated activity |
| `sender_avg_amount` | Average amount sent by this account | Establishes a personal baseline |
| `sender_std_amount` | Standard deviation of sent amounts | Low std + sudden spike = anomaly |
| `sender_unique_recv` | Number of unique recipients | Fan-out: one sender → many recipients, classic layering signal |
| `sender_total_amount` | Total volume sent | Structuring detection |
| `receiver_tx_count` | Total transactions into this receiver | High-frequency receiving is a red flag |
| `receiver_unique_send` | Number of unique senders | Fan-in: many senders → one receiver, common in scam collection accounts |
| `amount_deviation` | (Amount − sender_avg) / sender_std | How unusual is this transaction for *this specific account*? |
| `currency_mismatch` | 1 if sent and received currency differ | Currency conversion is a core layering technique |
| `cross_border` | 1 if sender and receiver banks are in different countries | Jurisdiction hopping to obscure audit trails |

### Anomaly Score
Isolation Forest is run on the full behavioural feature set to produce a continuous `anomaly_score` per transaction. This score is included as an input feature to the supervised classifiers — acting as an unsupervised pre-signal layer.

---

## Pipeline

```
Raw CSV (9.5M rows)
    │
    ├── Data Cleaning (null check)
    │
    ├── Feature Engineering
    │     ├── Temporal: hour, day, month, is_weekend, is_night
    │     └── Behavioural: velocity, fan-out, fan-in, deviation, cross_border, currency_mismatch
    │
    ├── Isolation Forest → anomaly_score
    │
    ├── Train/Test Split (80/20, stratified)
    │
    ├── Preprocessing (StandardScaler + OneHotEncoder)
    │
    ├── VarianceThreshold (removes near-zero-variance features)
    │
    ├── SMOTE (training set only — prevents leakage)
    │
    ├── Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV on 2% tuning subset)
    │
    ├── Model Training (LR, RF, AdaBoost, GaussianNB)
    │
    ├── Evaluation (Precision, Recall, F1, ROC AUC, PR AUC, Confusion Matrices, PR Curves)
    │
    └── Alert Demo (structured output per flagged transaction)
```

---

## Models

| Model | Role | Notes |
|---|---|---|
| Logistic Regression | Interpretable baseline | `class_weight='balanced'`, `solver='saga'` |
| Random Forest | Main ensemble model | `class_weight='balanced'`, feature importances available |
| AdaBoost | Boosted ensemble | Learns from misclassifications iteratively |
| GaussianNB | Fast probabilistic baseline | Dense input required; no `class_weight` support |

**Primary metric:** F1 Score (balances precision and recall for the minority class)  
**Secondary metric:** PR AUC (more reliable than ROC AUC on heavily imbalanced data)

---

## Project Structure

```
SafeFlow_AI.ipynb             # Main notebook
safeflow_best_model.joblib    # Saved best classifier (generated after training)
safeflow_preprocessor.joblib  # Saved ColumnTransformer (generated after training)
safeflow_selector.joblib      # Saved VarianceThreshold selector (generated after training)
README.md
```

---

## Sample Alert Output

```
=======================================================
         SAFEFLOW AI — TRANSACTION ALERT
=======================================================
  Actual Label    : ⚠️  SUSPICIOUS
  Predicted Label : ⚠️  SUSPICIOUS
  Risk Score      : 0.8732
-------------------------------------------------------
  TRANSACTION DETAILS
  Date            : 2022-11-21
  Time            : 02:14:07
  Amount          : UK pounds 156,892.69
  Payment Type    : ACH
  Sender Bank     : UK
  Receiver Bank   : UAE
  Currency Change : Yes ⚠️
  Cross-border    : Yes ⚠️
  Time of Day     : Night ⚠️
  Weekend         : No
-------------------------------------------------------
  BEHAVIOURAL CONTEXT
  Sender TX Count : 3
  Sender Avg Amt  : 4,210.00
  Amount Deviation: +36.24x avg
  Unique Recipients: 3 (fan-out)
  Unique Senders  : 14 (fan-in)
=======================================================
```

> Note: values above are illustrative. Actual outputs depend on the transaction sampled.

---

## Requirements

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib scipy
```

A Kaggle account and API credentials (`kaggle.json`) are required to download the dataset via `kagglehub`.

---

## Usage

1. Clone the repository and install dependencies
2. Configure Kaggle API credentials (`~/.kaggle/kaggle.json`)
3. Open `SafeFlow_AI.ipynb` in Jupyter Notebook or JupyterLab
4. Run cells sequentially — feature engineering must complete before anomaly detection
5. After training, `safeflow_best_model.joblib`, `safeflow_preprocessor.joblib`, and `safeflow_selector.joblib` will be saved to your working directory

---

## Notes

- `Laundering_type` is dropped before training to prevent label leakage
- `Sender_account` and `Receiver_account` are used only to compute aggregate features, then dropped from the model feature matrix
- `GaussianNB` requires dense arrays; sparse matrices are converted with `.toarray()` before prediction
- Behavioural features are computed on the full dataset before splitting — this is appropriate because they are historical aggregates of transaction counts and amounts, not derived from the target label
- SMOTE is applied only to the training set to avoid contaminating the test set with synthetic samples

## Results

The models were evaluated on a held-out 20% test set containing approximately **1.9 million transactions**.

### Model Performance Comparison

| Model               | Precision | Recall | F1 Score | ROC AUC | PR AUC |
| ------------------- | --------- | ------ | -------- | ------- | ------ |
| Random Forest       | 0.9899    | 0.9914 | 0.9906   | 0.9999  | 0.9955 |
| AdaBoost            | 0.9899    | 0.9914 | 0.9906   | 0.9998  | 0.9938 |
| Logistic Regression | 0.0111    | 0.8785 | 0.0218   | 0.9671  | 0.1748 |
| GaussianNB          | 0.0050    | 0.6709 | 0.0098   | 0.8592  | 0.0069 |

### Best Model

**Random Forest** achieved the highest F1 score and was selected as the production model.

| Metric    | Score  |
| --------- | ------ |
| Precision | 98.99% |
| Recall    | 99.14% |
| F1 Score  | 99.06% |
| ROC AUC   | 99.99% |
| PR AUC    | 99.55% |

The model correctly identified the vast majority of suspicious transactions while generating very few false positives.

### Random Forest Confusion Matrix

|                   | Predicted Normal | Predicted Suspicious |
| ----------------- | ---------------: | -------------------: |
| Actual Normal     |        1,898,976 |                   20 |
| Actual Suspicious |               17 |                1,958 |

This means:

* Only **20 legitimate transactions** were incorrectly flagged.
* Only **17 suspicious transactions** were missed.
* Over **99% recall** was achieved on the minority class.

### Key Findings

#### Random Forest

* Best overall performance.
* Highest PR AUC and ROC AUC.
* Excellent balance between precision and recall.
* Chosen as the final deployed model.

#### AdaBoost

* Nearly identical performance to Random Forest.
* Strong alternative ensemble approach.
* Slightly lower PR AUC.

#### Logistic Regression

* Very high recall (87.85%).
* Extremely low precision (1.11%).
* Generated a large number of false alerts.
* Useful as an interpretable baseline model.

#### Gaussian Naive Bayes

* Lowest overall performance.
* High false-positive rate.
* Demonstrated that simple probabilistic assumptions are insufficient for complex behavioural fraud patterns.

### Example Alert

The selected Random Forest model successfully identified the following suspicious transaction:

```text
Actual Label    : SUSPICIOUS
Predicted Label : SUSPICIOUS
Risk Score      : 1.0000

Amount          : UK pounds 11,112.95
Payment Type    : ACH
Sender Bank     : UK
Receiver Bank   : UK

Currency Change : No
Cross-border    : No
Time of Day     : Day

Sender TX Count : 179
Sender Avg Amt  : 9,498.16
Amount Deviation: +0.11x avg
Unique Recipients: 21 (fan-out)
Unique Senders  : 14 (fan-in)
```

Although the transaction was not cross-border and did not involve currency conversion, the behavioural patterns surrounding the accounts were sufficient for the model to classify it as suspicious. This demonstrates the value of behavioural feature engineering beyond simple rule-based detection.
