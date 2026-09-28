# Credit Card Fraud Detection with Anomaly Detection

A complete, local-first Python project for exploring rare credit-card fraud with unsupervised anomaly detection. It includes an interactive Streamlit dashboard, a command-line training workflow, reproducible synthetic demo data, and a saved model/predictions export.

## Quick start in VS Code

1. Open this folder in VS Code (`File → Open Folder…`).
2. Open the integrated terminal and create/activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

3. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

   The app opens in your browser. Choose **Use built-in synthetic demo data** to explore immediately, or upload a CSV.

## Dataset

The project accepts the widely used European cardholder dataset commonly named `creditcard.csv` (with `Time`, `Amount`, anonymized `V1`…`V28`, and optional binary `Class`; `Class=1` means fraud). The dataset is not bundled. Obtain it from the source where you have access, then either upload it in the dashboard or place it at `data/creditcard.csv` for the CLI.

You may also use a CSV with numeric transaction features and an optional `Class` column. The label is never used to fit the anomaly detector; it is used only for evaluation. If no label exists, the app reports anomaly scores and review candidates without fraud metrics.

## Run from the command line

```bash
python -m src.train --data data/creditcard.csv --model isolation_forest
```

Use `--model lof` to select Local Outlier Factor. Results are written to `outputs/`: a metrics JSON, scored transaction CSV, and reusable fitted model bundle. To produce and train on a small synthetic dataset:

```bash
python -m src.make_demo_data --output data/demo_transactions.csv
python -m src.train --data data/demo_transactions.csv
```

Synthetic data is only for checking the workflow and demonstrating the UI; it is not representative evidence of real-world fraud performance.

## What the model does

- Splits labeled data into train and test partitions when both classes permit it.
- Fits an unsupervised detector using **only legitimate training transactions** when labels are available. Without labels, it fits on all rows.
- Scales numeric features, excluding `Class` from model inputs.
- Scores held-out transactions and selects a review threshold from the configured contamination rate.
- Reports precision, recall, F1, average precision, ROC AUC (where defined), and a confusion matrix when test labels exist.
- Provides Isolation Forest and novelty-mode Local Outlier Factor.

Anomaly detection ranks unusual activity; unusual does not automatically mean fraudulent. Review thresholds should be chosen with operational capacity and labeled validation data in mind. This is an educational starting point, not a production payment decision system.

## Project layout

```text
app.py                  Streamlit dashboard
src/data.py             CSV validation and demo data
src/model.py            preprocessing, fitting, scoring, evaluation
src/train.py            command-line workflow and exports
src/make_demo_data.py   reproducible synthetic CSV generator
data/                   local datasets (ignored by git)
outputs/                generated model and result files
```
