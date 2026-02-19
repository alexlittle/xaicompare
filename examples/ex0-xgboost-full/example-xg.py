# ------------------------------------------------------------
# XGBoost + TF-IDF + SHAP on clinical notes (multi-class ready)
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# -------------------------
# 0) Example data loading
# -------------------------
# Replace this with your own data source. The only expectation is:
#   df["note"] : str, the clinical note text
#   df["label"]: target diagnosis (int or str)
# For demo, we’ll create a tiny synthetic dataframe. Replace with your dataset read.
df = pd.DataFrame({
    "note": [
        "Patient reports chest pain radiating to left arm; troponin elevated; ECG shows ST elevation.",
        "Fever and productive cough; chest x-ray suggests consolidation; likely pneumonia.",
        "Frequent urination and thirst; fasting glucose high; HbA1c elevated; diabetes mellitus suspected.",
        "Shortness of breath on exertion; BNP elevated; lower extremity edema; heart failure exacerbation.",
        "Sore throat, runny nose, mild fever; COVID test negative; likely viral upper respiratory infection."
    ],
    "label": ["myocardial_infarction", "pneumonia", "diabetes", "heart_failure", "viral_uri"]
})

# Optional: if your labels are strings, scikit-learn will handle them, but you can encode if you prefer.
X = df["note"].values
y = df["label"].values

# -------------------------
# 1) Train/validation split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -------------------------
# 2) Build pipeline
# -------------------------
# TfidfVectorizer defaults can be tuned:
#   - ngram_range=(1,2) to include bigrams
#   - min_df / max_df to prune rare/common terms
#   - max_features to control dimensionality
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=2,           # bump to 2+ for larger corpora; keep 1 for small samples
    max_df=0.9,
    strip_accents="unicode",
    lowercase=True
)

# XGBoost params: keep them modest for demo; tune later if needed.
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective='multi:softprob' if len(np.unique(y)) > 2 else 'binary:logistic',
    random_state=42,
    n_jobs=-1,
    tree_method="hist"  # 'gpu_hist' if you have GPU
)

# Fit TF-IDF + XGB in a single pipeline for convenience
pipe = Pipeline([
    ("tfidf", vectorizer),
    ("xgb", xgb)
])

pipe.fit(X_train, y_train)

# -------------------------
# 3) Evaluate
# -------------------------
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# -------------------------
# 4) SHAP: explain AFTER model is built
# -------------------------
# We need the *trained* vectorizer and model:
tfidf_fitted = pipe.named_steps["tfidf"]
xgb_fitted   = pipe.named_steps["xgb"]

# Transform the text to the sparse TF-IDF matrix that XGB saw:
X_train_tfidf = tfidf_fitted.transform(X_train)
X_test_tfidf  = tfidf_fitted.transform(X_test)

# Create a SHAP explainer tailored for tree models
# NOTE: use feature_perturbation="interventional" (default in recent SHAP)
explainer = shap.TreeExplainer(xgb_fitted, feature_perturbation="interventional")
# For multi-class: shap_values is a list of arrays, one per class
shap_values = explainer.shap_values(X_test_tfidf)

# -------------------------
# 5) Map feature indices → tokens
# -------------------------
feature_names = tfidf_fitted.get_feature_names_out()

# -------------------------
# 6) Global feature importance (SHAP)
# -------------------------
# For binary: shap_values is (n_samples, n_features)
# For multi-class: choose a class, or aggregate across classes.
def plot_global_importance_for_class(class_index=None, top_n=20):
    """
    For multi-class:
        class_index = integer for the class you want to inspect
    For binary:
        class_index = None
    """
    if isinstance(shap_values, list):
        # Multi-class: select the class’ SHAP matrix (n_samples, n_features)
        sv = shap_values[class_index]
        title = f"Global SHAP feature importance (class = {xgb_fitted.classes_[class_index]})"
    else:
        sv = shap_values
        title = "Global SHAP feature importance (binary)"
    # Mean absolute SHAP per feature
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    top_features = feature_names[top_idx]
    top_importances = mean_abs[top_idx]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_features))[::-1], top_importances[::-1], color="#4e79a7")
    plt.yticks(range(len(top_features))[::-1], top_features[::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example usage:
if isinstance(shap_values, list):
    # Plot for the most frequent class in y_train (or pick any)
    most_common_class = pd.Series(y_train).value_counts().index[0]
    class_idx = list(xgb_fitted.classes_).index(most_common_class)
    plot_global_importance_for_class(class_index=class_idx, top_n=15)
else:
    plot_global_importance_for_class(top_n=15)

# -------------------------
# 7) Local explanation for a single note
# -------------------------
i = 0  # index within X_test
x_row = X_test_tfidf[i]

# For multi-class, choose the class to explain (typically the predicted class)
if isinstance(shap_values, list):
    pred_proba = xgb_fitted.predict_proba(x_row)
    pred_class_idx = int(np.argmax(pred_proba))
    sv_row = shap_values[pred_class_idx][i].toarray().ravel() if hasattr(shap_values[pred_class_idx], "toarray") else shap_values[pred_class_idx][i]
    label_name = xgb_fitted.classes_[pred_class_idx]
else:
    sv_row = shap_values[i].toarray().ravel() if hasattr(shap_values, "toarray") else shap_values[i]
    label_name = xgb_fitted.classes_[1] if len(xgb_fitted.classes_) == 2 else xgb_fitted.classes_[0]

# Extract top contributing tokens for this instance
k = 15
top_idx_local = np.argsort(np.abs(sv_row))[-k:][::-1]
tokens_local = feature_names[top_idx_local]
contrib_local = sv_row[top_idx_local]

print("\n--- Local explanation for one note ---")
print("Original text:\n", X_test[i])
print(f"\nPredicted class: {label_name}")
print("\nTop token contributions (token -> SHAP value):")
for tok, val in zip(tokens_local, contrib_local):
    sign = "+" if val >= 0 else "-"
    print(f"{tok:30s} {sign}{abs(val):.4f}")

# Optional: SHAP force or waterfall plot (works well in notebooks)
# shap.initjs()
# if isinstance(shap_values, list):
#     shap.force_plot(explainer.expected_value[pred_class_idx], sv_row, matplotlib=True)
# else:
#     shap.force_plot(explainer.expected_value, sv_row, matplotlib=True)
# plt.show()