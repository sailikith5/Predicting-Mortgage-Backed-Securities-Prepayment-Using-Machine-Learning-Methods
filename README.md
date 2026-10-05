# Predicting Mortgage-Backed Securities Prepayment Using Machine Learning

UE24CS352A Machine Learning mini-project (Section E, Problem #55).
**Team:** Sai Likhith S (PES1UG24AM239) and Srivamshi Chikkireddy (PES1UG24AM287).

Loan-level prediction of whether a mortgage prepays in a given month, following Guo, Boccio & Cameron,
*Predicting Mortgage Backed Securities Prepayment Using Machine Learning Methods* (Stanford CS229, 2021), plus
extensions: gradient boosting, a stacked ensemble, an out-of-time split, a leakage ablation, confidence intervals and a
balance-weighted pool-level (CPR) evaluation. The 2-page write-up is `MBS_Prepayment_Writeup.pdf`.

## Problem
Binary classification at loan-month level: `label = 1` if the loan prepays in full that month, else `0`.
With the five vintages used here, about 1.25% of loan-months are prepayments (roughly 79:1 imbalance).
Models: logistic regression (plain/L1/L2), SVM (RBF/poly/sigmoid), Gaussian discriminant analysis (linear/quadratic),
a feed-forward neural network, gradient boosting and a stacked ensemble.

## Data (not committed: licence and size)
- **Freddie Mac Single Family Loan-Level sample dataset** (free, needs registration):
  https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset. We used the vintages 2005, 2008, 2012, 2016, 2019.
- **Macro data** from FRED (downloaded by `src/get_macro.py`): 30-year mortgage rate, unemployment, FHFA house-price index.
- `data/`, `models/` and trained artefacts are git-ignored; they are rebuilt by the steps below.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run (from the project root, in order)
Put the unzipped Freddie Mac files anywhere under `data/raw/`, keeping the original names:
`sample_orig_YYYY.txt` with `sample_perf_YYYY.txt` (or `sample_svcg_YYYY.txt`), or for the full dataset
`historical_data_YYYYQn.txt` with `historical_data_time_YYYYQn.txt`.

```bash
python src/get_macro.py                                          # -> data/macro.csv
python src/data_prep.py --loans-per-file 5000 --split vintage    # -> data/processed/data.parquet
python src/eda.py                                                # -> results/eda/
python src/train.py --tune                                       # leak-free (main result) -> results/leakfree
python src/train.py --tune --paper-mode                          # paper setup with the leak -> results/paper
python src/predict.py --tag leakfree --n 10                      # live demo on sample test loans
```
Flags: `--tune` (hyper-parameter search on validation data), `--boot 200` (bootstrap resamples, 0 = off),
`--max-train N` (cap training rows), `--skip-svm`, `--skip-nn`, `--epochs N`, `--l1-rows N` (L1 logistic is trained on
at most N rows because liblinear is slow). Dry run without real data: `python tests/make_synthetic.py`.

## Method summary
- Origination + performance files merged on loan ID; macro series joined by month; one-hot categoricals; training
  medians for missing values; standardisation with training statistics only.
- **Out-of-time split** (`--split vintage`): oldest 60% of loans (by origination month) train, next 20% validation, newest
  20% test. No loan is in two splits. (`--split loan` gives the paper's random split by loan ID.)
- **Features:** rate incentive (loan rate minus market rate), refinance ratio, loan age, credit score, LTV/DTI, balance
  ratio, delinquency, macro variables, seasonality, and backward-looking history: incentive 3 and 6 months ago,
  months spent in the money, maximum past incentive (burnout), and 3/6/12-month market-rate change.
- **Imbalance:** accuracy is misleading at 79:1, so each model is reported at the 0.5 cut-off and at a cut-off tuned for F1
  on the validation set (the test set is never used for tuning), with TPR/TNR, precision, F1, ROC-AUC, PR-AUC and
  lift in the top 1% of scores. SVMs use `class_weight="balanced"`.
- **Tuning (`--tune`):** L2 strength and a random search over gradient-boosting settings, chosen by validation PR-AUC.
- **Ensemble:** logistic-regression stacker fitted on validation-set logits of LogReg-L2, QDA, boosting and the network.
- **Confidence intervals:** loan-level bootstrap on the test set (`results/<mode>/ci.csv`), including the paired difference
  between the ensemble and boosting.
- **Pool-level evaluation:** monthly prepayment rate (`pool_level.*`) and balance-weighted SMM/CPR (`pool_cpr.*`) of
  predictions against actuals on test loans. CPR counts full payoffs only on a sample of loans, not a full pool.
- **Leakage ablation:** the paper's top feature `current_upb` is zero in the payoff month and reveals the label.
  `--paper-mode` reproduces that setup; the default uses the previous month's balance.
- Neural network: in-128-256-1, BatchNorm + ReLU, SGD with decaying learning rate, `pos_weight=10`, best-validation checkpoint.

## Results
Results are written to `results/leakfree/` and `results/paper/` (`metrics.csv`, `ci.csv`, `tuning.json`, `top_features.csv`,
confusion matrices, PR curves, pool-level plots). The write-up summarises them. Paper-mode scores of about 1.00 are
the leak, not skill; the leak-free numbers are the real performance.
