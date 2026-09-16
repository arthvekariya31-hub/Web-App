"""
app.py
------
Flask prototype web application for the dissertation:
"Evaluating the Effectiveness of Machine Learning Techniques for
Detecting AI-Generated Phishing Emails".

Loads the best-performing, cross-validated and hyperparameter-tuned model
(selected by train.py) and lets a user paste email text or upload a
.txt/.eml file to get a phishing/legitimate prediction with a confidence
score and the top contributing words. The homepage also presents the full
research findings: model comparison, cross-validation results, the
stylometric-feature ablation study (RQ2), and the cross-origin
generalisation analysis (RQ3/RQ4 proxy).

Run with:  python3 app.py
Then open: http://127.0.0.1:5000
"""

import email
import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request

from preprocessing import clean_text, extract_stylometric_features, estimate_ai_writing_style

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model_artifacts"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB upload limit

# ---- Load trained model + vectorizer once at startup ----
model = joblib.load(MODEL_DIR / "best_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
stylo_scaler = joblib.load(MODEL_DIR / "stylo_scaler.joblib")
MODEL_NAME = (MODEL_DIR / "best_model_name.txt").read_text().strip()
FEATURE_MODE = (MODEL_DIR / "feature_mode.txt").read_text().strip()  # "tfidf_only" or "tfidf_plus_stylometric"
with open(MODEL_DIR / "training_results.json") as f:
    RESULTS = json.load(f)

FEATURE_NAMES = np.array(vectorizer.get_feature_names_out())
STYLO_KEYS = ["word_count", "exclamation_count", "url_count", "urgency_word_count",
              "first_person_ratio", "lexical_diversity"]


def parse_eml(file_bytes: bytes):
    msg = email.message_from_bytes(file_bytes)
    subject = msg.get("subject", "") or ""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if ctype == "text/plain" and "attachment" not in disp:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode(charset, errors="replace")
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        body = payload.decode(charset, errors="replace") if payload else msg.get_payload()
    return subject, body


def build_feature_vector(subject: str, body: str):
    """Build the feature vector matching FEATURE_MODE, exactly mirroring
    how train.py constructed features at training time."""
    cleaned = clean_text(subject, body)
    vec_tfidf = vectorizer.transform([cleaned])
    if FEATURE_MODE == "tfidf_only":
        return vec_tfidf, cleaned
    stylo = extract_stylometric_features(subject, body)
    stylo_arr = np.array([[stylo[k] for k in STYLO_KEYS]])
    stylo_scaled = stylo_scaler.transform(stylo_arr)
    from scipy import sparse
    combined = sparse.hstack([vec_tfidf, sparse.csr_matrix(stylo_scaled)]).tocsr()
    return combined, cleaned


def top_contributing_words(cleaned_text: str, top_n: int = 8):
    if not hasattr(model, "coef_"):
        return []
    vec = vectorizer.transform([cleaned_text])
    nonzero_idx = vec.nonzero()[1]
    if len(nonzero_idx) == 0:
        return []
    n_tfidf_features = len(FEATURE_NAMES)
    coefs = model.coef_[0][:n_tfidf_features]  # ignore any appended stylometric coefficients
    contributions = vec[0, nonzero_idx].toarray().flatten() * coefs[nonzero_idx]
    order = np.argsort(-np.abs(contributions))[:top_n]
    words = FEATURE_NAMES[nonzero_idx][order]
    contribs = contributions[order]
    return [
        {"word": w, "direction": "phishing" if c > 0 else "legitimate", "weight": round(float(c), 3)}
        for w, c in zip(words, contribs)
    ]


def predict(subject: str, body: str):
    vec, cleaned = build_feature_vector(subject, body)
    pred = int(model.predict(vec)[0])
    confidence = None
    phishing_probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = float(proba[pred])
        phishing_probability = float(proba[1])  # raw P(phishing), regardless of predicted class

    ai_style = estimate_ai_writing_style(subject, body)
    ai_style_fraction = ai_style["score"] / 100.0

    # Combined, single headline metric requested by the user: "how likely is
    # this an AI-generated phishing email" -- both conditions must hold, so
    # we combine as a product of the two independent signals:
    #   P(phishing)  x  P(AI-style writing)
    # This is a simple, transparent combination rule (not a trained joint
    # classifier) -- documented as such in the README/dissertation.
    combined_score = round(phishing_probability * ai_style_fraction * 100, 1) if phishing_probability is not None else None

    return {
        "label": "Phishing" if pred == 1 else "Legitimate",
        "is_phishing": bool(pred == 1),
        "confidence": round(confidence * 100, 1) if confidence is not None else None,
        "phishing_probability": round(phishing_probability * 100, 1) if phishing_probability is not None else None,
        "combined_score": combined_score,
        "top_words": top_contributing_words(cleaned),
        "stylometric": extract_stylometric_features(subject, body),
        "ai_style": ai_style,
        "cleaned_preview": cleaned[:300],
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", model_name=MODEL_NAME, feature_mode=FEATURE_MODE, results=RESULTS)


@app.route("/analyze", methods=["POST"])
def analyze():
    subject, body = "", ""
    input_mode = request.form.get("input_mode", "paste")
    error = None

    if input_mode == "paste":
        subject = request.form.get("subject", "").strip()
        body = request.form.get("body", "").strip()
        if not body:
            error = "Please paste some email text before analysing."
    else:
        uploaded = request.files.get("email_file")
        if not uploaded or uploaded.filename == "":
            error = "Please choose a .txt or .eml file to upload."
        else:
            filename = uploaded.filename.lower()
            raw = uploaded.read()
            if filename.endswith(".eml"):
                subject, body = parse_eml(raw)
            else:
                body = raw.decode("utf-8", errors="replace")
                subject = ""

    if error:
        return render_template("index.html", model_name=MODEL_NAME, feature_mode=FEATURE_MODE, results=RESULTS, error=error)

    result = predict(subject, body)
    return render_template(
        "index.html", model_name=MODEL_NAME, feature_mode=FEATURE_MODE, results=RESULTS,
        prediction=result, submitted_subject=subject, submitted_body=body,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
