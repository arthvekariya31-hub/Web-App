"""
train.py (v3, checkpointed)
----------------------------
Same rigorous pipeline as before (cross-validation, hyperparameter tuning,
ablation study, subset/cross-origin analysis, XGBoost, plots), but broken
into checkpointed stages so it can be safely resumed if interrupted, and
sized for a constrained single-core environment. Hyperparameter search
spaces are intentionally small (documented in the dissertation as a
resource-constrained search) rather than exhaustive.

Run with:  python3 train.py
Safe to re-run repeatedly -- each stage is skipped if its checkpoint
already exists, so it resumes automatically. Prints "ALL STAGES COMPLETE"
when finished.
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_validate,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

from preprocessing import clean_text, extract_stylometric_features

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "model_artifacts"
PLOT_DIR = BASE / "static" / "plots"
CACHE = BASE / "cache"
for d in (MODEL_DIR, PLOT_DIR, CACHE):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SEARCH_SUBSAMPLE = 3000
CV_FOLDS = 3
SEARCH_CV_FOLDS = 3
MAX_FEATURES = 8000

MODEL_NAMES = ["Logistic Regression", "Random Forest", "Naive Bayes", "XGBoost"]
COLORS = {"Logistic Regression": "#2e5395", "Random Forest": "#2f7d4f",
          "Naive Bayes": "#b23b3b", "XGBoost": "#c98a1f"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def done(name):
    return (CACHE / name).exists()


def load_json(name):
    with open(CACHE / name) as f:
        return json.load(f)


def save_json(name, obj):
    with open(CACHE / name, "w") as f:
        json.dump(obj, f, indent=2)


# ============================== STAGE 1: prepare data ======================
def stage_prepare():
    if done("prepared.flag"):
        log("Stage 1 (prepare) already done, skipping.")
        return
    log("Stage 1: loading + cleaning dataset...")
    files = [
        "Combined-Fine-tune-Nonphishing.csv", "Combined-Fine-tune-phishing.csv",
        "Prompt_nonPhishing_10000.csv", "Prompt_phishing_10000.csv",
    ]
    frames = []
    for fname in files:
        d = pd.read_csv(DATA_DIR / fname)
        d.columns = [c.lower() for c in d.columns]
        d = d[["label", "subject", "body"]].copy()
        d["source_file"] = fname
        d["origin"] = "fine_tune" if fname.startswith("Combined-Fine-tune") else "prompt_engineered"
        frames.append(d)
    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["body"])
    full["subject"] = full["subject"].fillna("")
    mask_amb = (full["source_file"] == "Combined-Fine-tune-phishing.csv") & (full["label"] == 0)
    full = full[~mask_amb]
    full = full.drop_duplicates(subset=["subject", "body"])
    full["label"] = full["label"].astype(int)
    full = full.reset_index(drop=True)
    log(f"  {len(full)} emails after cleaning. Label dist: {full['label'].value_counts().to_dict()}")

    log("  cleaning text (regex tokeniser + stemming)...")
    full["clean_text"] = full.apply(lambda r: clean_text(r["subject"], r["body"]), axis=1)
    full = full[full["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)

    log("  extracting stylometric features...")
    stylo = pd.DataFrame(list(full.apply(lambda r: extract_stylometric_features(r["subject"], r["body"]), axis=1)))

    strat_key = full["label"].astype(str) + "_" + full["origin"]
    idx_train, idx_test = train_test_split(full.index, test_size=0.2, stratify=strat_key, random_state=RANDOM_STATE)
    train_df = full.loc[idx_train].reset_index(drop=True)
    test_df = full.loc[idx_test].reset_index(drop=True)
    stylo_train = stylo.loc[idx_train].reset_index(drop=True)
    stylo_test = stylo.loc[idx_test].reset_index(drop=True)

    log("  fitting TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr_tfidf = vectorizer.fit_transform(train_df["clean_text"])
    Xte_tfidf = vectorizer.transform(test_df["clean_text"])
    log(f"  vocab size: {len(vectorizer.vocabulary_)}")

    scaler = MinMaxScaler()
    stylo_tr_scaled = scaler.fit_transform(stylo_train.values)
    stylo_te_scaled = scaler.transform(stylo_test.values)
    Xtr_combined = sparse.hstack([Xtr_tfidf, sparse.csr_matrix(stylo_tr_scaled)]).tocsr()
    Xte_combined = sparse.hstack([Xte_tfidf, sparse.csr_matrix(stylo_te_scaled)]).tocsr()

    joblib.dump(Xtr_tfidf, CACHE / "Xtr_tfidf.joblib")
    joblib.dump(Xte_tfidf, CACHE / "Xte_tfidf.joblib")
    joblib.dump(Xtr_combined, CACHE / "Xtr_combined.joblib")
    joblib.dump(Xte_combined, CACHE / "Xte_combined.joblib")
    joblib.dump(train_df["label"].values, CACHE / "y_train.joblib")
    joblib.dump(test_df["label"].values, CACHE / "y_test.joblib")
    joblib.dump(train_df["origin"].values, CACHE / "origin_train.joblib")
    joblib.dump(test_df["origin"].values, CACHE / "origin_test.joblib")
    joblib.dump(vectorizer, CACHE / "vectorizer.joblib")
    joblib.dump(scaler, CACHE / "stylo_scaler.joblib")

    meta = {
        "dataset_size": int(len(full)),
        "label_distribution": {str(k): int(v) for k, v in full["label"].value_counts().to_dict().items()},
        "origin_distribution": {str(k): int(v) for k, v in full["origin"].value_counts().to_dict().items()},
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }
    save_json("meta.json", meta)
    (CACHE / "prepared.flag").write_text("ok")
    log("Stage 1 complete.")


# =========================== STAGE 2: hyperparameter tuning ================
PARAM_SPACES = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                              {"C": [0.1, 0.5, 1, 2, 5], "penalty": ["l2"]}, True, 5),
    "Naive Bayes": (MultinomialNB(), {"alpha": [0.05, 0.1, 0.5, 1.0]}, False, None),
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                        {"n_estimators": [100, 150], "max_depth": [20, 30], "min_samples_leaf": [1, 2]}, True, 3),
    "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, tree_method="hist", n_jobs=1),
                 {"n_estimators": [100, 150], "max_depth": [4, 6], "learning_rate": [0.1, 0.2]}, True, 3),
}


def stage_tune(model_name):
    ckpt = f"best_params_{model_name.replace(' ', '_')}.json"
    if done(ckpt):
        log(f"Stage 2 ({model_name} tuning) already done, skipping.")
        return
    if not done("prepared.flag"):
        log("Data not prepared yet, run stage_prepare first.")
        return
    log(f"Stage 2: tuning {model_name}...")
    Xtr_tfidf = joblib.load(CACHE / "Xtr_tfidf.joblib")
    y_train = joblib.load(CACHE / "y_train.joblib")

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(y_train), size=min(SEARCH_SUBSAMPLE, len(y_train)), replace=False)
    Xs, ys = Xtr_tfidf[idx], y_train[idx]

    estimator, params, use_random, n_iter = PARAM_SPACES[model_name]
    cv = StratifiedKFold(n_splits=SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.time()
    if use_random:
        search = RandomizedSearchCV(estimator, params, n_iter=n_iter, cv=cv, scoring="f1",
                                     random_state=RANDOM_STATE, n_jobs=1)
    else:
        search = GridSearchCV(estimator, params, cv=cv, scoring="f1", n_jobs=1)
    search.fit(Xs, ys)
    log(f"  {model_name} best params: {search.best_params_} (CV F1={search.best_score_:.4f}, {time.time()-t0:.1f}s)")
    save_json(ckpt, {"best_params": search.best_params_, "cv_f1": search.best_score_})
    log(f"Stage 2 ({model_name}) complete.")


# =========================== STAGE 3: CV + final fit ========================
def build_model(model_name, params):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **params)
    if model_name == "Random Forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
    if model_name == "Naive Bayes":
        return MultinomialNB(**params)
    if model_name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, tree_method="hist", n_jobs=1, **params)
    raise ValueError(model_name)


def full_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)), "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(), "n": int(len(y_true)),
    }


def stage_cv_and_fit(model_name):
    ckpt = f"cvfit_{model_name.replace(' ', '_')}.json"
    if done(ckpt):
        log(f"Stage 3 ({model_name} CV+fit) already done, skipping.")
        return
    params_ckpt = f"best_params_{model_name.replace(' ', '_')}.json"
    if not done(params_ckpt):
        log(f"{model_name} not tuned yet, run stage_tune first.")
        return
    log(f"Stage 3: {CV_FOLDS}-fold CV + final fit for {model_name}...")
    Xtr_tfidf = joblib.load(CACHE / "Xtr_tfidf.joblib")
    Xte_tfidf = joblib.load(CACHE / "Xte_tfidf.joblib")
    y_train = joblib.load(CACHE / "y_train.joblib")
    y_test = joblib.load(CACHE / "y_test.joblib")
    params = load_json(params_ckpt)["best_params"]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.time()
    scores = cross_validate(build_model(model_name, params), Xtr_tfidf, y_train, cv=cv,
                             scoring=["accuracy", "f1", "roc_auc"], n_jobs=1)
    cv_result = {
        "accuracy_mean": float(scores["test_accuracy"].mean()), "accuracy_std": float(scores["test_accuracy"].std()),
        "f1_mean": float(scores["test_f1"].mean()), "f1_std": float(scores["test_f1"].std()),
        "roc_auc_mean": float(scores["test_roc_auc"].mean()), "roc_auc_std": float(scores["test_roc_auc"].std()),
    }
    log(f"  CV done in {time.time()-t0:.1f}s. F1={cv_result['f1_mean']:.4f} (+/-{cv_result['f1_std']:.4f})")

    t0 = time.time()
    model = build_model(model_name, params)
    model.fit(Xtr_tfidf, y_train)
    y_pred = model.predict(Xte_tfidf)
    y_proba = model.predict_proba(Xte_tfidf)[:, 1]
    test_result = full_metrics(y_test, y_pred, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    log(f"  final fit done in {time.time()-t0:.1f}s. Test F1={test_result['f1']:.4f} Acc={test_result['accuracy']:.4f}")

    joblib.dump(model, CACHE / f"model_{model_name.replace(' ', '_')}.joblib")
    joblib.dump((fpr.tolist(), tpr.tolist()), CACHE / f"roc_{model_name.replace(' ', '_')}.joblib")
    save_json(ckpt, {"cv_result": cv_result, "test_result": test_result})
    log(f"Stage 3 ({model_name}) complete.")


# =========================== STAGE 4: ablation + subset + cross-origin =====
def stage_extra_analysis():
    if done("extra_analysis.json"):
        log("Stage 4 (extra analysis) already done, skipping.")
        return
    if not all(done(f"cvfit_{n.replace(' ', '_')}.json") for n in MODEL_NAMES):
        log("Not all models CV+fit done yet.")
        return
    log("Stage 4: ablation study + subset breakdown + cross-origin generalisation...")

    test_results = {n: load_json(f"cvfit_{n.replace(' ', '_')}.json")["test_result"] for n in MODEL_NAMES}
    best_name = max(test_results, key=lambda n: test_results[n]["f1"])
    log(f"  Best model (TF-IDF only) by test F1: {best_name}")

    Xtr_tfidf = joblib.load(CACHE / "Xtr_tfidf.joblib")
    Xte_tfidf = joblib.load(CACHE / "Xte_tfidf.joblib")
    Xtr_combined = joblib.load(CACHE / "Xtr_combined.joblib")
    Xte_combined = joblib.load(CACHE / "Xte_combined.joblib")
    y_train = joblib.load(CACHE / "y_train.joblib")
    y_test = joblib.load(CACHE / "y_test.joblib")
    origin_train = joblib.load(CACHE / "origin_train.joblib")
    origin_test = joblib.load(CACHE / "origin_test.joblib")
    params = load_json(f"best_params_{best_name.replace(' ', '_')}.json")["best_params"]
    best_model_tfidf = joblib.load(CACHE / f"model_{best_name.replace(' ', '_')}.joblib")

    # --- Ablation: TF-IDF vs TF-IDF+stylometric ---
    log("  Ablation (RQ2): TF-IDF vs TF-IDF+stylometric...")
    model_combined = build_model(best_name, params)
    model_combined.fit(Xtr_combined, y_train)
    y_pred_c = model_combined.predict(Xte_combined)
    y_proba_c = model_combined.predict_proba(Xte_combined)[:, 1]
    combined_results = full_metrics(y_test, y_pred_c, y_proba_c)
    f1_gain = combined_results["f1"] - test_results[best_name]["f1"]
    use_combined = f1_gain > 0.003
    log(f"    TF-IDF only F1={test_results[best_name]['f1']:.4f} | +stylometric F1={combined_results['f1']:.4f} "
        f"| gain={f1_gain:+.4f} | adopt_combined={use_combined}")

    # --- Subset breakdown by origin (RQ3/4 proxy) ---
    log("  Subset breakdown by origin...")
    subset_results = {}
    for origin in ["fine_tune", "prompt_engineered"]:
        mask = origin_test == origin
        if mask.sum() == 0:
            continue
        yp = best_model_tfidf.predict(Xte_tfidf[mask])
        ypr = best_model_tfidf.predict_proba(Xte_tfidf[mask])[:, 1]
        subset_results[origin] = full_metrics(y_test[mask], yp, ypr)
        log(f"    {origin}: n={subset_results[origin]['n']} acc={subset_results[origin]['accuracy']:.4f} f1={subset_results[origin]['f1']:.4f}")

    # --- Cross-origin generalisation ---
    log("  Cross-origin generalisation...")
    cross_origin_results = {}
    for train_o, test_o in [("fine_tune", "prompt_engineered"), ("prompt_engineered", "fine_tune")]:
        trm = origin_train == train_o
        tem = origin_test == test_o
        m = build_model(best_name, params)
        m.fit(Xtr_tfidf[trm], y_train[trm])
        yp = m.predict(Xte_tfidf[tem])
        ypr = m.predict_proba(Xte_tfidf[tem])[:, 1]
        key = f"train_{train_o}_test_{test_o}"
        cross_origin_results[key] = full_metrics(y_test[tem], yp, ypr)
        log(f"    {key}: n={cross_origin_results[key]['n']} acc={cross_origin_results[key]['accuracy']:.4f} f1={cross_origin_results[key]['f1']:.4f}")

    final_model = model_combined if use_combined else best_model_tfidf
    final_feature_mode = "tfidf_plus_stylometric" if use_combined else "tfidf_only"
    final_test_metrics = combined_results if use_combined else test_results[best_name]

    joblib.dump(final_model, CACHE / "final_model.joblib")
    save_json("extra_analysis.json", {
        "best_model": best_name,
        "final_feature_mode": final_feature_mode,
        "final_test_metrics": final_test_metrics,
        "ablation": {"model": best_name, "tfidf_only": test_results[best_name],
                     "tfidf_plus_stylometric": combined_results, "f1_gain": f1_gain, "adopted_for_app": use_combined},
        "subset_breakdown_by_origin": subset_results,
        "cross_origin_generalisation": cross_origin_results,
    })
    log("Stage 4 complete.")


# =========================== STAGE 5: plots + final save ====================
def stage_finalize():
    if done("finalized.flag"):
        log("Stage 5 (finalize) already done, skipping.")
        return
    if not done("extra_analysis.json"):
        log("Run stage_extra_analysis first.")
        return
    log("Stage 5: generating plots and saving final artifacts...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = load_json("meta.json")
    extra = load_json("extra_analysis.json")
    test_results = {n: load_json(f"cvfit_{n.replace(' ', '_')}.json")["test_result"] for n in MODEL_NAMES}
    cv_results = {n: load_json(f"cvfit_{n.replace(' ', '_')}.json")["cv_result"] for n in MODEL_NAMES}
    best_params = {n: load_json(f"best_params_{n.replace(' ', '_')}.json")["best_params"] for n in MODEL_NAMES}
    best_name = extra["best_model"]

    # Confusion matrix
    cm = np.array(extra["final_test_metrics"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legitimate", "Phishing"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}\n({extra['final_feature_mode']})", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "confusion_matrix.png", dpi=140); plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name in MODEL_NAMES:
        fpr, tpr = joblib.load(CACHE / f"roc_{name.replace(' ', '_')}.joblib")
        ax.plot(fpr, tpr, label=f"{name} (AUC={test_results[name]['roc_auc']:.3f})", color=COLORS.get(name), linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models (TF-IDF, test set)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "roc_curves.png", dpi=140); plt.close(fig)

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = MODEL_NAMES
    f1s = [test_results[n]["f1"] for n in names]
    bars = ax.bar(names, f1s, color=[COLORS.get(n) for n in names])
    ax.set_ylim(0.80, 1.0); ax.set_ylabel("F1-score (test set)")
    ax.set_title("Model Comparison (TF-IDF features)", fontsize=11)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=15)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "model_comparison.png", dpi=140); plt.close(fig)

    # Top features (Logistic Regression coefficients)
    lr_model = joblib.load(CACHE / "model_Logistic_Regression.joblib")
    vectorizer = joblib.load(CACHE / "vectorizer.joblib")
    feat_names = np.array(vectorizer.get_feature_names_out())
    coefs = lr_model.coef_[0]
    top_pos = np.argsort(-coefs)[:12]
    top_neg = np.argsort(coefs)[:12]
    idx = np.concatenate([top_neg[::-1], top_pos])
    vals = coefs[idx]; labels = feat_names[idx]
    colors_bar = ["#b23b3b" if v > 0 else "#2f7d4f" for v in vals]
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.barh(range(len(vals)), vals, color=colors_bar)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression coefficient (red = pushes toward phishing)")
    ax.set_title("Most Influential TF-IDF Terms", fontsize=11)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "top_features.png", dpi=140); plt.close(fig)

    # Save final model artifacts
    final_model = joblib.load(CACHE / "final_model.joblib")
    joblib.dump(final_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
    joblib.dump(joblib.load(CACHE / "stylo_scaler.joblib"), MODEL_DIR / "stylo_scaler.joblib")
    (MODEL_DIR / "best_model_name.txt").write_text(best_name)
    (MODEL_DIR / "feature_mode.txt").write_text(extra["final_feature_mode"])

    summary = {
        **meta,
        "search_subsample_size": SEARCH_SUBSAMPLE,
        "cv_folds": CV_FOLDS,
        "best_hyperparameters": best_params,
        "cross_validation_results": cv_results,
        "best_model": best_name,
        "final_feature_mode": extra["final_feature_mode"],
        "results": test_results,
        "ablation_tfidf_plus_stylometric": extra["ablation"],
        "subset_breakdown_by_origin": extra["subset_breakdown_by_origin"],
        "cross_origin_generalisation": extra["cross_origin_generalisation"],
    }
    save_json("training_results_final.json", summary)
    with open(MODEL_DIR / "training_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    (CACHE / "finalized.flag").write_text("ok")
    log("Stage 5 complete. ALL STAGES COMPLETE")


def main():
    stage_prepare()
    for n in MODEL_NAMES:
        stage_tune(n)
    for n in MODEL_NAMES:
        stage_cv_and_fit(n)
    stage_extra_analysis()
    stage_finalize()


if __name__ == "__main__":
    main()
