"""EDA plots -> results/eda/"""
import os

import matplotlib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.makedirs("results/eda", exist_ok=True)
df = pd.read_parquet("data/processed/data.parquet")
print("rows", len(df), "loans", df["loan_id"].nunique())
print("class balance:\n", df["label"].value_counts().rename({0: "no prepay", 1: "prepay"}))
print("imbalance ratio ~", round((df["label"] == 0).sum() / max((df["label"] == 1).sum(), 1)), ": 1")

ax = df["label"].value_counts().sort_index().plot.bar(color=["#4c72b0", "#dd8452"])
ax.set_xticklabels(["No prepay", "Prepay"], rotation=0); ax.set_title("Class balance")
plt.tight_layout(); plt.savefig("results/eda/class_balance.png", dpi=150); plt.close()

fig, axs = plt.subplots(2, 2, figsize=(9, 6))
for ax, col, q in zip(axs.ravel(), ["rate_incentive_cur", "credit_score", "orig_ltv", "loan_age"], [20, 10, 10, 20]):
    b = pd.qcut(df[col], q=q, duplicates="drop")
    g = df.groupby(b, observed=True)["label"].mean() * 100
    ax.plot(range(len(g)), g.values, marker="o"); ax.set_title(f"Prepay rate (%) vs {col}")
    ax.set_xlabel(f"{col} (quantile bins)")
plt.tight_layout(); plt.savefig("results/eda/prepay_rate_vs_features.png", dpi=150); plt.close()

num = df.select_dtypes("number").drop(columns=["label"]).fillna(0)
corr = num.corrwith(df["label"]).dropna().sort_values()
pd.concat([corr.head(8), corr.tail(8)]).plot.barh(figsize=(6, 5), title="Correlation with prepay label")
plt.tight_layout(); plt.savefig("results/eda/label_correlation.png", dpi=150); plt.close()

pos = df[df["label"] == 1].sample(min(500, int(df["label"].sum())), random_state=0)
neg = df[df["label"] == 0].sample(1500, random_state=0)
s = pd.concat([pos, neg])
Xs = s.select_dtypes("number").drop(columns=["label"]).fillna(0)
Z = PCA(3).fit_transform(StandardScaler().fit_transform(Xs))
fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(111, projection="3d")
m = s["label"].values == 1
ax.scatter(*Z[~m].T, c="red", s=6, label="No prepayment", alpha=.5)
ax.scatter(*Z[m].T, c="blue", s=6, label="Prepayment", alpha=.5)
ax.set_title("PCA, 3 components"); ax.legend()
plt.tight_layout(); plt.savefig("results/eda/pca3.png", dpi=150); plt.close()
print("saved plots to results/eda/")
