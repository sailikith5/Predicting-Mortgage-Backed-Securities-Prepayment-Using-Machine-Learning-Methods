# Predicting MBS Prepayment with Machine Learning

UE24CS352A Machine Learning mini-project. Loan-level prediction of whether a mortgage
prepays in a given month, replicating and extending Guo, Boccio & Cameron,
*Predicting Mortgage Backed Securities Prepayment Using Machine Learning Methods* (Stanford CS229, 2021).

**Team:** <Name 1, SRN> · <Name 2, SRN> · Section <X> · Problem # <N>

## Problem
Binary classification at loan-month level: `label = 1` if the loan prepays in that month, else `0`
(about 50:1 class imbalance). Models: Logistic Regression (none/L1/L2), SVM (RBF/poly/sigmoid),
Gaussian Discriminant Analysis (linear/quadratic), Feed-forward Neural Network, plus two extensions:
HistGradientBoosting and a soft-vote Ensemble.

## Data
- **Freddie Mac Single Family Loan-Level Dataset** (origination + monthly performance files). Free, needs registration:
  https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
- **Macro data** (FRED): 30-yr mortgage rate, unemployment rate, FHFA house price index (YoY appreciation).
- Raw data is NOT committed (see `.gitignore`).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run (in order)
```bash
# 1. Put Freddie Mac files under data/raw/ (unzipped .txt files):
#    sample_orig_YYYY.txt + sample_svcg_YYYY.txt   (sample dataset), or
#    historical_data_YYYYQn.txt + historical_data_time_YYYYQn.txt   (full dataset)

python src/get_macro.py                       # -> data/macro.csv
python src/data_prep.py --loans-per-file 3000 # -> data/processed/data.parquet
python src/eda.py                             # -> results/eda/*.png

python src/train.py --paper-mode              # paper setup (uses current_upb)   -> results/paper, models/paper
python src/train.py                           # leak-free (prev-month balance)   -> results/leakfree, models/leakfree

python src/predict.py --tag leakfree --n 10   # demo on sample test loans
```
Useful flags: `--max-train 500000` (faster), `--skip-svm`, `--skip-nn`, `--epochs 10`.

Stricter, out-of-time evaluation (recommended as the headline result): train on older origination
vintages, test on the newest ones.
```bash
python src/data_prep.py --loans-per-file 3000 --split vintage
python src/train.py
```

Dry run without real data: `python tests/make_synthetic.py`, then
`python src/data_prep.py --raw data/raw_synth --macro data/macro_synth.csv`.

## Method summary
- Origination + performance merged on loan ID; macro variables joined by month; one-hot categoricals; missing values
  imputed with training medians; features standardized (train statistics only).
- Split **by loan ID** 60/20/20 so no loan appears in two splits.
- Features include rate incentive (loan rate minus market rate), loan age, credit score, LTV/CLTV/DTI, balance
  features, unemployment, house-price appreciation.
- Neural net: in→128→256→1, BatchNorm+ReLU, SGD lr 0.1 decayed x0.91/epoch, 30 epochs, BCE with `pos_weight=10`,
  best-validation-loss checkpoint.
- SVMs are trained on a subsample (kernel cost grows quadratically) and evaluated on a test subsample.
- **Imbalance:** accuracy is meaningless at ~50:1, so every model is reported at the default 0.5 cut-off *and* at a
  cut-off tuned for F1 on the **validation** set (test set never used for tuning), with TPR/TNR, precision, F1,
  ROC-AUC, PR-AUC and precision/lift in the top 1% of scores. SVMs use `class_weight="balanced"`.
- **Ensemble:** logistic-regression stacker fitted on validation-set logits of LogReg-L2, QDA, HistGB and the NN.
- **Pool-level check (MBS view):** average predicted prepay probability per calendar month vs the actual monthly
  prepay rate on test loans (`results/<mode>/pool_level.png`, correlation + MAE printed).
- **Extra features:** refinance ratio (loan rate / market rate), 3-month change in mortgage rate, month-of-year
  seasonality (sin/cos), alongside rate incentive, age, credit, LTV/DTI, balance ratio, unemployment, HPA.
- **Leakage check:** the paper's top feature is `current_upb`, which may reveal the label (balance is ~0 in the payoff
  month). `--paper-mode` reproduces the paper; the default replaces it with previous-month balance.

## Results
Fill from `results/<mode>/metrics.csv` (use the `val-F1` rows; accuracy, TPR/TNR, precision, recall, F1, ROC-AUC, PR-AUC,
precision_top1pct, lift_top1pct). Expect `paper` mode to look near-perfect: that is the `current_upb` leak, not skill.

| Model | Accuracy | TPR | TNR | F1 | PR-AUC |
|---|---|---|---|---|---|
| ... | | | | | |

## Repo layout
```
src/get_macro.py   download macro series        src/train.py    train + evaluate all models
src/data_prep.py   merge + features + split     src/predict.py  demo inference
src/eda.py         EDA + PCA plots              src/models.py   PyTorch network
tests/make_synthetic.py  fake data for a dry run (numbers from it are NOT results)
```
