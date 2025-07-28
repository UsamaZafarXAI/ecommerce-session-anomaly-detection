import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
from scipy.stats import mannwhitneyu
import lightgbm as lgb
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
import shap
import lime.lime_tabular

warnings.filterwarnings("ignore", category=FutureWarning, module="shap")

# Paths
BASE_PATH  = "/kaggle/input/ecommerce-dataset"
OUTPUT_DIR = "/kaggle/working/retailrocket_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
events = pd.read_csv(f"{BASE_PATH}/events.csv")
events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")
events.sort_values(["visitorid", "timestamp"], inplace=True)

gap = events.groupby("visitorid")["timestamp"].diff().gt(pd.Timedelta(minutes=30))
first = events["visitorid"].ne(events["visitorid"].shift())
events["new_session"] = (gap | first).astype(int)
events["session_idx"] = events.groupby("visitorid")["new_session"].cumsum()
events["session_id"]  = events["visitorid"].astype(str)+"_"+events["session_idx"].astype(str)

# Feature engineering
session_df = events.groupby("session_id").agg(
    session_duration=("timestamp", lambda x: (x.max()-x.min()).total_seconds()),
    n_events=("event", "count"),
    n_views=("event", lambda x: (x=="view").sum()),
    n_cart=("event", lambda x: (x=="addtocart").sum()),
    n_transaction=("event", lambda x: (x=="transaction").sum()),
    unique_items=("itemid", pd.Series.nunique)
).reset_index()

session_df["cart_view_ratio"] = session_df["n_cart"] / (session_df["n_views"] + 1e-6)
session_df["purchase_view_ratio"] = session_df["n_transaction"] / (session_df["n_views"] + 1e-6)
session_df["event_rate"] = session_df["n_events"] / (session_df["session_duration"] + 1e-6)

features = session_df[[
    "session_duration", "n_events", "n_views", "n_cart",
    "n_transaction", "unique_items",
    "cart_view_ratio", "purchase_view_ratio", "event_rate"
]].fillna(0)

feature_names = features.columns.tolist()
X_scaled = StandardScaler().fit_transform(features)

# Anomaly detection: Isolation Forest and Autoencoder
iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_labels = (iso.fit_predict(X_scaled) == -1).astype(int)

input_dim = X_scaled.shape[1]
inp = Input(shape=(input_dim,))
enc = Dense(8, activation="tanh", activity_regularizer=regularizers.l1(1e-4))(inp)
enc = Dense(4, activation="relu")(enc)
dec = Dense(8, activation="tanh")(enc)
dec = Dense(input_dim, activation="linear")(dec)
ae = Model(inp, dec)
ae.compile("adam", "mse")
ae.fit(X_scaled, X_scaled, epochs=10, batch_size=64, validation_split=0.1, verbose=0)

recon = ae.predict(X_scaled, verbose=0)
mse = np.mean(np.square(X_scaled - recon), axis=1)
ae_labels = (mse > np.percentile(mse, 95)).astype(int)

final_labels = ((iso_labels + ae_labels) >= 1).astype(int)
session_df["label"] = final_labels

# Supervised classification
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, final_labels, test_size=0.2,
                                          stratify=final_labels, random_state=42)
clf = lgb.LGBMClassifier(random_state=42)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)

pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value": [accuracy_score(y_te, y_pred),
              precision_score(y_te, y_pred),
              recall_score(y_te, y_pred),
              f1_score(y_te, y_pred)]
}).to_csv(f"{OUTPUT_DIR}/classification_metrics.csv", index=False)

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

# SHAP explanations
explainer = shap.Explainer(clf)
shap_vals = explainer(X_te)[:, :, 1]
shap.summary_plot(shap_vals, X_te, feature_names, show=False)
plt.savefig(f"{OUTPUT_DIR}/shap_summary.png")
plt.close()
shap.plots.bar(shap_vals, show=False)
plt.savefig(f"{OUTPUT_DIR}/shap_bar.png")
plt.close()

for i, idx in enumerate(np.where(y_pred == 1)[0][:3], 1):
    shap.plots.waterfall(shap_vals[idx], max_display=10, show=False)
    plt.savefig(f"{OUTPUT_DIR}/shap_waterfall_{i}.png")
    plt.close()

# LIME explanations
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_tr,
    feature_names=feature_names,
    class_names=["Normal", "Anomalous"],
    mode="classification"
)
for i in range(3):
    fig = lime_exp.explain_instance(X_te[i], clf.predict_proba,
                                    num_features=len(feature_names)).as_pyplot_figure()
    plt.savefig(f"{OUTPUT_DIR}/lime_session_{i + 1}.png")
    plt.close(fig)

# Statistical validation: Cliff's delta
def cliffs_delta(a, b):
    m, n = len(a), len(b)
    U, _ = mannwhitneyu(a, b, method="asymptotic", alternative="two-sided")
    delta = (2 * U) / (m * n) - 1
    ad = abs(delta)
    size = ("negligible" if ad < 0.147 else
            "small" if ad < 0.33 else
            "medium" if ad < 0.474 else
            "large")
    return delta, size

def delta_ci(x, y, B=1000, alpha=0.05, rng=None, max_sample=5000):
    rng = np.random.default_rng(rng)
    nx0, ny0 = len(x), len(y)
    take = min(max_sample, nx0, ny0)
    deltas = []
    for _ in range(B):
        xs = rng.choice(x, take, replace=True)
        ys = rng.choice(y, take, replace=True)
        deltas.append(cliffs_delta(xs, ys)[0])
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi

norm_df = session_df[session_df.label == 0]
anom_df = session_df[session_df.label == 1]
n_bal = min(len(norm_df), len(anom_df))
norm_s = norm_df.sample(n_bal, random_state=42)
anom_s = anom_df.sample(n_bal, random_state=42)

rows = []
for col in feature_names:
    delta, size = cliffs_delta(anom_s[col], norm_s[col])
    lo, hi = delta_ci(anom_s[col].values, norm_s[col].values, B=1000)
    rows.append({
        "Feature": col,
        "Cliffs_delta": round(delta, 3),
        "CI_95%": f"[{lo:.3f}, {hi:.3f}]",
        "Effect": size
    })

pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/statistical_validation.csv", index=False)

# Visualization
plt.figure(figsize=(15, 12))
for i, col in enumerate(feature_names):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=session_df, x="label", y=col, palette="Set2")
    plt.title(col.replace("_", " "))
    plt.xlabel("")
    plt.xticks([0, 1], ["Normal", "Anomalous"])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_boxplots.png")
plt.close()

corr = features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_correlation.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.hist((iso_labels == -1).astype(int) + ae_labels, bins=np.arange(3) - 0.5,
         edgecolor="black", rwidth=0.8)
plt.title("Anomaly Ensemble Score")
plt.xlabel("Number of Models Flagging Anomaly")
plt.ylabel("Session Count")
plt.xticks([0, 1, 2])
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/anomaly_score_hist.png")
plt.close()

session_df.to_csv(f"{OUTPUT_DIR}/session_features_with_labels.csv", index=False)
