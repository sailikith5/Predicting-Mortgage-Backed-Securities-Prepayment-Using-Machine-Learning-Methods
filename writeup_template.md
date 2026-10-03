# Predicting Mortgage-Backed Securities Prepayment Using Machine Learning
**UE24CS352A Machine Learning, Mini-Project** · Team: <Name 1 (SRN)>, <Name 2 (SRN)> · Section <X> · Problem #<N>
Repo: <GitHub URL>

## 1. Problem Statement
Prepayments reduce the interest income of mortgage-backed securities investors, so predicting them matters for pricing
and risk. We model prepayment at the individual loan level: given a loan's origination characteristics, current state
and macroeconomic conditions, predict whether it prepays in a given month (binary classification). Based on
Guo, Boccio & Cameron (Stanford CS229, 2021).

## 2. Dataset
Freddie Mac Single Family Loan-Level Dataset (origination + monthly performance), joined with macro data from FRED/FHFA/BLS
(mortgage rate, unemployment, house-price appreciation). We sampled <N> loans per <vintage/file> giving <rows> loan-months
from <years>, with <F> features after one-hot encoding. Only <p>% of rows are prepayments (about <ratio>:1 imbalance).
Split 60/20/20 (train/val/test) by loan ID; headline results use an out-of-time split (oldest vintages train, newest test).

## 3. Approach
Preprocessing: sentinel codes to missing, median imputation, standardization, one-hot encoding. Key features: rate
incentive, loan age, credit score, LTV/DTI, balance ratio, macro variables. Models: logistic regression (L1/L2), SVM
(RBF/poly/sigmoid, subsample), GDA (linear/quadratic), feed-forward neural net (in-128-256-1, pos_weight 10), plus
gradient boosting and a soft-vote ensemble. Because of the class imbalance we report TPR/TNR, precision, F1 and
PR-AUC alongside accuracy, and tune each model's decision threshold on the validation set (not the test set). The ensemble is
a stacked logistic regression. We also tested a leakage-free variant that replaces current balance with previous-month balance,
and compared predicted vs actual monthly prepayment rates at pool level.

## 4. Implementation Overview
Python (pandas, scikit-learn, PyTorch). `data_prep.py` builds the loan-month table, `train.py` trains and evaluates all
models and saves metrics, plots and models, `predict.py` runs the demo. All steps are reproducible via the README.

## 5. Results
<Insert metrics table from results/paper/metrics.csv and results/leakfree/metrics.csv, one confusion matrix figure.>

| Model | Acc | TPR | TNR | F1 | PR-AUC |
|---|---|---|---|---|---|
| | | | | | |

Top features (L1 logistic): <list top 5>.

## 6. Conclusions
<Best model and why. Effect of removing current_upb. Accuracy inflated by imbalance. Limitations: subsample, SVM on subset,
no survival model. Future work: UPB-weighted SMM/CPR comparison on full pools, survival analysis, ensemble tuning.>
