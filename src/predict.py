"""Demo: load saved models and predict on sample test loans.
  python src/predict.py --tag leakfree --n 10
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="leakfree", choices=["leakfree", "paper"])
    ap.add_argument("--data", default="data/processed/data.parquet")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    md = f"models/{a.tag}"
    meta = json.load(open(f"{md}/meta.json"))
    scaler = joblib.load(f"{md}/scaler.joblib")
    df = pd.read_parquet(a.data)
    te = df[df["split"] == "test"]
    pos = te[te["label"] == 1].sample(min(a.n // 2, int(te["label"].sum())), random_state=a.seed)
    neg = te[te["label"] == 0].sample(a.n - len(pos), random_state=a.seed)
    s = pd.concat([pos, neg]).sample(frac=1, random_state=a.seed)

    X = s[meta["features"]].astype("float32").fillna(pd.Series(meta["medians"]))
    X = scaler.transform(X).astype("float32")

    out = pd.DataFrame({"loan_id": s["loan_id"].values,
                        "month": s["period"].dt.strftime("%Y-%m").values,
                        "ACTUAL": s["label"].values})
    probs = {}
    for f in sorted(glob.glob(f"{md}/*.joblib")):
        name = os.path.basename(f)[:-7]
        if name == "scaler" or name.startswith("_"):
            continue
        m = joblib.load(f)
        probs[name] = m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.predict(X).astype(float)
    if meta.get("has_nn"):
        import torch
        from models import PrepayNet, predict_nn
        net = PrepayNet(meta["in_dim"])
        net.load_state_dict(torch.load(f"{md}/NeuralNet.pt"))
        probs["NeuralNet"] = predict_nn(net, X)
    stack = joblib.load(f"{md}/_stack.joblib")
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    probs["Ensemble"] = stack.predict_proba(np.column_stack([lg(probs[k]) for k in meta["ensemble"]]))[:, 1]
    thr = meta["thresholds"]["Ensemble"]
    for k in ["LogReg_L2", "GDA_Quadratic", "HistGB", "SVM_rbf", "NeuralNet", "Ensemble"]:
        if k in probs:
            out[k] = np.round(probs[k], 3)
    out["PRED(Ensemble)"] = (probs["Ensemble"] >= thr).astype(int)
    pd.set_option("display.width", 200)
    print(f"\nPrepayment probabilities ({a.tag} models); ACTUAL: 1 = prepaid this month\n")
    print(out.to_string(index=False))
    pred = (probs["Ensemble"] >= thr).astype(int)
    print(f"\nEnsemble (cut-off {thr:.3f}, tuned on validation set) correct on {(pred == out['ACTUAL'].values).sum()}/{len(out)} sample loans")


if __name__ == "__main__":
    main()
