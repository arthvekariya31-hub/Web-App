# AI-Generated Phishing Detection Prototype

A Flask web application demonstrating machine learning-based detection of
phishing emails (both conventional and AI-generated), built for the MSc
Cyber Security dissertation *"Evaluating the Effectiveness of Machine
Learning Techniques for Detecting AI-Generated Phishing Emails"*.

## Dataset

This project uses the **israksu/PhishingDataset** (shared by the
dissertation supervisor): https://github.com/israksu/PhishingDataset

> "This dataset is a synthetic set of phishing emails generated using
> prompt engineering and LLM fine-tuning."

**Important note for your Methodology/Limitations chapter:** both the
phishing *and* the "legitimate" emails in this dataset are AI-generated —
there is no real human-written phishing in it. This means the app and
results below answer *"can ML tell AI-generated phishing apart from
AI-generated legitimate email?"*, not *"AI-generated vs human-written
phishing"* as in the original research questions. Discuss this with your
supervisor: either adjust the RQs to match, or add a human-written
phishing set (e.g. SpamAssassin/Nazario) alongside this one.

Four CSV files are combined (included in `data/`):
- `Combined-Fine-tune-Nonphishing.csv` / `Combined-Fine-tune-phishing.csv`
  — emails from a **fine-tuned** generation model
- `Prompt_nonPhishing_10000.csv` / `Prompt_phishing_10000.csv`
  — emails from **prompt-engineered** generation

After cleaning (removing empty bodies, duplicates, and ~10 mislabelled
rows), **37,869 emails** remain (19,430 phishing / 18,439 legitimate).

## What's included

```
webapp/
├── data/                     # the 4 dataset CSVs
├── preprocessing.py          # shared text-cleaning used by training + app
├── train.py                  # checkpointed training pipeline (resumable)
├── model_artifacts/          # saved model, vectorizer, scaler, full results JSON
├── static/plots/             # confusion matrix, ROC curves, comparison chart, feature importances
├── app.py                    # Flask application
├── templates/                # HTML templates
├── static/                   # CSS + JS
├── requirements.txt
└── README.md                 # this file
```

## How to run it yourself

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Retrain the models.** A trained model is already included
   in `model_artifacts/`, so this step is optional. If you want to
   retrain from scratch:
   ```bash
   python3 train.py
   ```
   The script is **checkpointed and resumable** — it saves progress to
   `cache/` after each stage (data prep, hyperparameter tuning per model,
   cross-validation + final fit per model, ablation/subset analysis,
   plots). If it's interrupted, or if you're on a slow/single-core
   machine, just run `python3 train.py` again — it picks up where it left
   off instead of starting over. On a normal multi-core laptop this
   should complete in a single run (a few minutes); it was originally
   developed on a constrained single-core environment, which is why the
   checkpointing exists. Delete the `cache/` folder to force a full
   retrain from scratch.

3. **Run the web app**:
   ```bash
   python3 app.py
   ```
   Then open **http://127.0.0.1:5000**

4. **Use it**: paste an email's subject/body, or upload a `.txt`/`.eml`
   file, and click "Analyse Email". The homepage also shows the full
   research findings (model comparison, cross-validation, ablation study,
   generalisation analysis) with charts.

## Real results obtained (for your dissertation Results chapter)

All numbers below are genuine outputs from this pipeline — nothing is
invented. Trained on 30,295 emails, evaluated on a held-out test set of
7,574 emails. TF-IDF features (unigrams+bigrams, 8,000 features),
hyperparameters selected via a resource-constrained randomised/grid search
(3-fold CV on an 3,000-row subsample, documented as such — a full
exhaustive grid search was not computationally feasible in the
development environment used).

### Model comparison (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 97.1% | 97.5% | 96.9% | 0.972 | 0.996 |
| Random Forest | 95.1% | — | — | 0.952 | 0.991 |
| XGBoost | 95.0% | — | — | 0.951 | 0.990 |
| Naive Bayes | 93.3% | — | — | 0.935 | 0.985 |

⭐ Logistic Regression selected as final model (best F1), tuned params:
`C=5, penalty='l2'`.

### 3-fold cross-validation (training set, stability check)

| Model | Accuracy (mean ± std) | F1 (mean ± std) |
|---|---|---|
| Logistic Regression | 0.968 ± 0.001 | 0.969 ± 0.001 |
| Random Forest | 0.946 ± 0.003 | 0.948 ± 0.003 |
| XGBoost | 0.947 ± 0.002 | 0.948 ± 0.002 |
| Naive Bayes | 0.933 ± 0.001 | 0.934 ± 0.001 |

Low standard deviations indicate the results are stable across folds, not
an artefact of one lucky split.

### RQ2 — Ablation study: does adding stylometric features help?

Adding the supplementary stylometric features (word count, URL count,
urgency-word count, first-person pronoun ratio, lexical diversity,
exclamation count) to Logistic Regression changed F1 from **0.9717**
(TF-IDF only) to **0.9716** (TF-IDF + stylometric) — a **negligible,
slightly negative** change. **The app uses TF-IDF features alone.**

This is a genuine, reportable finding: on this dataset, hand-crafted
stylometric features add nothing beyond what TF-IDF n-grams already
capture. Worth discussing honestly in your Discussion chapter — it does
not mean stylometric features are useless in general (other studies, e.g.
Opara, Modesti & Golightly (2025), found them valuable), but that TF-IDF
alone was already sufficient for this specific dataset and feature set.

### RQ3/RQ4 — Generalisation across generation methods

The dataset contains two distinct generation methods: a **fine-tuned**
model and **prompt-engineered** generation. This gives a proxy for
cross-generator generalisation (the exact concern raised by Chen et al.,
2026):

| Scenario | n | Accuracy | F1 |
|---|---|---|---|
| Within-domain — tested on fine-tune subset (trained on both) | 3,996 | 95.6% | 0.956 |
| Within-domain — tested on prompt-engineered subset (trained on both) | 3,578 | 98.8% | 0.989 |
| **Cross-domain** — train on fine-tune, test on prompt-engineered | 3,578 | 90.7% | 0.910 |
| **Cross-domain** — train on prompt-engineered, test on fine-tune | 3,996 | **80.0%** | **0.801** |

**This is the most interesting real finding in the whole project.**
Performance drops substantially — from ~95–99% within-domain to as low
as **80%** — when the model is trained on emails from one generation
method and tested on the other. This is direct, dataset-internal
evidence of the generalisation gap that Chen et al. (2026) describe at a
literature-review level: a detector's high reported accuracy does not
necessarily transfer to phishing produced by a different generation
process. This finding should anchor your Discussion chapter.

## New: single combined "AI-Generated Phishing Likelihood" headline

The results page now leads with **one clear number**: e.g. *"54.4% — Possibly
an AI-generated phishing email."* This combines two underlying signals:

```
combined_score = P(phishing) × P(AI-generated writing style)
```

- **P(phishing)** comes from the trained, cross-validated Logistic
  Regression model (labelled "ML model" in the UI).
- **P(AI-generated writing style)** comes from the heuristic described
  below (labelled "Heuristic" in the UI).

Both underlying scores are still shown separately underneath the headline,
each clearly tagged with its source, so a technical reader (your
supervisor/examiner) can see exactly how the headline number was derived
and that the two components have very different levels of rigour. This
"headline number + transparent breakdown" design is worth describing
explicitly in your Chapter 4 (Implementation) — it's a reasonable UX
decision for a non-technical end user, while still being honest about
methodology for an academic audience.

## "AI-Generated Writing Style" indicator (heuristic, not ML)

Following supervisor/user feedback that users want to understand *how AI-generated*
an email looks (not just phishing/legitimate), the app now also shows a second,
clearly-labelled score: **"AI-Generated Writing Style Estimate"**.

**This is a transparent, rule-based heuristic, not a trained ML classifier.**
It could not be trained as a proper classifier because — as noted above — this
dataset has no real human-written emails to validate against. Instead, it scores
text (0–100%) against stylistic markers reported in the literature review
(Chapter 2): formulaic LLM stock phrases ("I hope this email finds you well",
"please do not hesitate to reach out"), unusually uniform sentence lengths,
low contraction usage, stable moderate-high lexical diversity, absence of
typos/informal markers, and em-dash usage. Each signal that fires is shown
to the user for transparency (see `preprocessing.py`,
`estimate_ai_writing_style()`).

**Be careful how you describe this in your dissertation** — call it an
"exploratory heuristic indicator", never a "detector" or "classifier", and
be explicit that it has not been empirically validated against ground-truth
labels (because no such labelled data was available). This is itself a
legitimate, honestly-reported limitation you can discuss: a natural next
step for future work would be to source or construct a genuine human-vs-AI
labelled dataset to properly train and validate this.

## Known limitations (be ready to discuss these with your supervisor)

1. **No human-written phishing/legitimate emails in this dataset** — see
   the note under "Dataset" above. This is the most important one to
   raise.
2. **Constrained hyperparameter search** — due to a single-core
   development environment, the search space and cross-validation folds
   were kept small (documented above). On your own machine with more
   compute, you can widen `PARAM_SPACES` and `CV_FOLDS`/`SEARCH_CV_FOLDS`
   in `train.py` for a more exhaustive search.
3. **Narrow "legitimate" email domain** — the legitimate class consists
   mostly of account-notification/security-alert-style emails rather
   than everyday correspondence, so genuinely unrelated legitimate
   emails may produce lower-confidence, borderline predictions.

## Notes on the pipeline (for writing up Chapter 4 — Implementation)

- **Preprocessing**: subject + body combined, HTML stripped, URLs/emails
  normalised to placeholder tokens, lower-cased, tokenised with a regex
  word tokeniser, English stop words removed (scikit-learn's built-in
  list), remaining tokens stemmed with the Porter stemmer.
- **Feature extraction**: TF-IDF, unigrams + bigrams, capped at 8,000
  features, minimum document frequency 3 (reduced from an initial 20,000
  to keep training feasible on constrained hardware — see Limitations).
- **Models**: Logistic Regression, Random Forest, Multinomial Naive
  Bayes, XGBoost — all hyperparameter-tuned via cross-validated
  search, then cross-validated (3-fold) and evaluated on a held-out
  20% test set, stratified by both label and generation-method origin.
- **Ablation study**: TF-IDF alone vs TF-IDF + stylometric features,
  directly answering RQ2 (see above).
- **Generalisation analysis**: within-domain vs cross-domain evaluation
  across the two generation methods present in the dataset, as a proxy
  for RQ3/RQ4 (see above).
- All figures (`static/plots/*.png`) are generated directly by
  `train.py` and can be dropped straight into your dissertation's
  Results chapter.
