"""Train + evaluate all models. Run twice for the leakage ablation:
  python src/train.py --paper-mode     # uses current_upb like the paper  -> results/paper
  python src/train.py                  # uses previous-month balance       -> results/leakfree

Imbalance handling: every model is reported at the default 0.5 cut-off AND at a cut-off chosen to maximise F1
on the VALIDATION set (the test set is never used for tuning). The ensemble is a logistic-regression stacker
fitted on validation-set probabilities of the base models.
"""
import argparse
import json
import os
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*l1_ratio.*")

ENSEMBLE = ["LogReg_L2", "GDA_Quadratic", "HistGB", "NeuralNet"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def best_f1_threshold(y, s):
    """Cut-off on score s that maximises F1 (used on validation data only)."""
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[np.argmax(f1)])


def report(name, y, score, thr, kind, rows, outdir, plot=False):
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    k = max(1, int(0.01 * len(y)))
    top = np.argsort(-score)[:k]
    p_top = float(y[top].mean())
    rows.append(dict(
        model=name, cutoff=kind, threshold=thr, n_test=len(y), accuracy=accuracy_score(y, pred),
        tnr=tn / max(tn + fp, 1), tpr=tp / max(tp + fn, 1),
        precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        roc_auc=roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan,
        pr_auc=average_precision_score(y, score) if y.sum() > 0 else np.nan,
        precision_top1pct=p_top, lift_top1pct=p_top / max(y.mean(), 1e-12),
    ))
    r = rows[-1]
    print(f"  {name:14s} [{kind:6s}] acc {r['accuracy']:.4f} tpr {r['tpr']:.3f} tnr {r['tnr']:.4f} "
          f"prec {r['precision']:.3f} f1 {r['f1']:.3f} pr_auc {r['pr_auc']:.3f} lift@1% {r['lift_top1pct']:.1f}x")
    if plot:
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No prepay", "Prepay"]); ax.set_yticklabels(["No prepay", "Prepay"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{name} (val-F1 cut-off)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"cm_{name}.png"), dpi=150)
        plt.close(fig)


def evaluate(name, y_te, s_te, y_va, s_va, rows, outdir, default_thr=0.5):
    """Report at the default cut-off and at the validation-tuned cut-off. Returns tuned threshold."""
    report(name, y_te, s_te, default_thr, "0.5" if default_thr == 0.5 else "0.0", rows, outdir)
    thr = best_f1_threshold(y_va, s_va)
    report(name, y_te, s_te, thr, "val-F1", rows, outdir, plot=True)
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/data.parquet")
    ap.add_argument("--paper-mode", action="store_true",
                    help="use current_upb (paper) instead of previous-month balance")
    ap.add_argument("--max-train", type=int, default=0, help="cap train rows (0 = all)")
    ap.add_argument("--svm-train", type=int, default=30000)
    ap.add_argument("--svm-test", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip-svm", action="store_true")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tag = "paper" if a.paper_mode else "leakfree"
    out_m, out_r = f"models/{tag}", f"results/{tag}"
    os.makedirs(out_m, exist_ok=True); os.makedirs(out_r, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    df = pd.read_parquet(a.data)
    drop = ["loan_id", "period", "label", "split"]
    drop += ["prev_upb", "upb_ratio_prev"] if a.paper_mode else ["current_upb", "upb_ratio_cur"]
    feats = [c for c in df.columns if c not in drop]
    print(f"mode={tag} features={len(feats)} rows={len(df):,}")

    Xdf = df[feats].astype("float32")
    tr_mask = (df["split"] == "train").values
    med = Xdf[tr_mask].median()
    Xdf = Xdf.fillna(med)
    scaler = StandardScaler().fit(Xdf[tr_mask])
    X = scaler.transform(Xdf).astype("float32")
    y = df["label"].values.astype("int8")
    idx_tr = np.where(tr_mask)[0]
    if a.max_train and len(idx_tr) > a.max_train:
        idx_tr = rng.choice(idx_tr, a.max_train, replace=False)
    idx_va = np.where((df["split"] == "val").values)[0]
    idx_te = np.where((df["split"] == "test").values)[0]
    Xtr, ytr, Xva, yva, Xte, yte = X[idx_tr], y[idx_tr], X[idx_va], y[idx_va], X[idx_te], y[idx_te]
    print(f"train {len(ytr):,} (pos {ytr.sum():,}) val {len(yva):,} (pos {yva.sum():,}) "
          f"test {len(yte):,} (pos {yte.sum():,})")

    ens = [k for k in ENSEMBLE if not (a.skip_nn and k == "NeuralNet")]
    joblib.dump(scaler, f"{out_m}/scaler.joblib")

    rows, probs, probs_va, thresholds = [], {}, {}, {}
    sk = {
        "LogReg": LogisticRegression(C=1e6, max_iter=300),
        "LogReg_L1": LogisticRegression(penalty="l1", solver="liblinear", C=1.0),
        "LogReg_L2": LogisticRegression(penalty="l2", C=1.0, max_iter=300),
        "GDA_Linear": LinearDiscriminantAnalysis(),
        "GDA_Quadratic": QuadraticDiscriminantAnalysis(reg_param=1e-3),
        "HistGB": HistGradientBoostingClassifier(max_iter=200, random_state=a.seed),
    }
    fitted = {}
    for name, m in sk.items():
        t = time.time()
        m.fit(Xtr, ytr)
        probs[name] = m.predict_proba(Xte)[:, 1]
        probs_va[name] = m.predict_proba(Xva)[:, 1]
        thresholds[name] = evaluate(name, yte, probs[name], yva, probs_va[name], rows, out_r)
        joblib.dump(m, f"{out_m}/{name}.joblib")
        fitted[name] = m
        print(f"    ({time.time()-t:.0f}s)")

    # top features from L1 logistic: |coef_raw| * mean(feature), as in the paper
    coef_std = fitted["LogReg_L1"].coef_[0]
    coef_raw = coef_std / scaler.scale_
    raw_mean = Xdf.iloc[idx_tr].mean().values
    imp = pd.DataFrame({"feature": feats, "coef_std": coef_std, "coef_raw": coef_raw,
                        "mean": raw_mean, "importance_paper": np.abs(coef_raw) * raw_mean,
                        "abs_coef_std": np.abs(coef_std)})
    imp.sort_values("abs_coef_std", ascending=False).to_csv(f"{out_r}/top_features.csv", index=False)
    print("  L1 kept", int((coef_std != 0).sum()), "of", len(feats), "features")

    if not a.skip_nn:
        import torch
        from models import predict_nn, train_nn
        print("neural network")
        net = train_nn(Xtr, ytr, Xva, yva, epochs=a.epochs, seed=a.seed)
        probs["NeuralNet"], probs_va["NeuralNet"] = predict_nn(net, Xte), predict_nn(net, Xva)
        thresholds["NeuralNet"] = evaluate("NeuralNet", yte, probs["NeuralNet"], yva, probs_va["NeuralNet"],
                                           rows, out_r)
        torch.save(net.state_dict(), f"{out_m}/NeuralNet.pt")

    # Ensemble = logistic-regression stacker on validation-set logits of the base models.
    # (A plain average is a poor fit: LogReg, QDA and the pos_weight-10 net are calibrated very differently.)
    stack = LogisticRegression(C=1.0, max_iter=500).fit(
        np.column_stack([logit(probs_va[k]) for k in ens]), yva)
    p_stack_va = stack.predict_proba(np.column_stack([logit(probs_va[k]) for k in ens]))[:, 1]
    p_stack = stack.predict_proba(np.column_stack([logit(probs[k]) for k in ens]))[:, 1]
    probs["Ensemble"] = p_stack
    thresholds["Ensemble"] = evaluate("Ensemble", yte, p_stack, yva, p_stack_va, rows, out_r)
    joblib.dump(stack, f"{out_m}/_stack.joblib")

    # Precision-recall curves (full-test models)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for k, pr in probs.items():
        if k in ("LogReg", "LogReg_L1", "GDA_Linear"):
            continue
        pc, rc, _ = precision_recall_curve(yte, pr)
        ax.plot(rc, pc, label=f"{k} (AP {average_precision_score(yte, pr):.3f})")
    ax.axhline(yte.mean(), color="gray", ls=":", label="no skill")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"PR curves ({tag})")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{out_r}/pr_curves.png", dpi=150); plt.close(fig)

    # Pool-level check (the MBS view): average predicted prepay probability per calendar month vs actual rate
    pool = pd.DataFrame({"period": df["period"].values[idx_te], "actual": yte, "pred": p_stack}) \
             .groupby("period").mean()
    if len(pool) >= 3:
        pool.to_csv(f"{out_r}/pool_level.csv")
        corr = float(pool["actual"].corr(pool["pred"]))
        mae = float((pool["actual"] - pool["pred"]).abs().mean())
        print(f"  pool-level (monthly SMM-style): corr {corr:.3f}  MAE {mae:.5f}  over {len(pool)} months")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(pool.index, pool["actual"] * 100, label="actual"); ax.plot(pool.index, pool["pred"] * 100, label="predicted")
        ax.set_ylabel("monthly prepay rate (%)"); ax.set_title(f"Pool-level, test loans ({tag})")
        ax.legend(); fig.tight_layout(); fig.savefig(f"{out_r}/pool_level.png", dpi=150); plt.close(fig)

    if not a.skip_svm:  # SVM on a subsample (kernel matrix grows quadratically); class-balanced
        s_tr = rng.choice(len(Xtr), min(a.svm_train, len(Xtr)), replace=False)
        s_va = rng.choice(len(Xva), min(a.svm_test // 2, len(Xva)), replace=False)
        s_te = rng.choice(len(Xte), min(a.svm_test, len(Xte)), replace=False)
        for kernel in ["rbf", "poly", "sigmoid"]:
            name = f"SVM_{kernel}"
            t = time.time()
            m = SVC(kernel=kernel, gamma="scale", class_weight="balanced").fit(Xtr[s_tr], ytr[s_tr])
            thresholds[name] = evaluate(name, yte[s_te], m.decision_function(Xte[s_te]),
                                        yva[s_va], m.decision_function(Xva[s_va]), rows, out_r, default_thr=0.0)
            joblib.dump(m, f"{out_m}/{name}.joblib")
            print(f"    ({time.time()-t:.0f}s)")

    json.dump({"features": feats, "medians": {k: float(v) for k, v in med.items()},
               "in_dim": len(feats), "ensemble": ens, "tag": tag, "has_nn": not a.skip_nn,
               "thresholds": thresholds}, open(f"{out_m}/meta.json", "w"), indent=1)

    res = pd.DataFrame(rows)
    res.to_csv(f"{out_r}/metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[res.cutoff != "0.5"].drop(columns=["n_test"]).round(4).to_string(index=False))
    print("saved", f"{out_r}/metrics.csv")


if __name__ == "__main__":
    main()
