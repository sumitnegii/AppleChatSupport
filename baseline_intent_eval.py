import pandas as pd
import numpy as np
import re
import html
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin


SEED = 42
TRAIN_FILE = 'golden_candidates.csv'
EVAL_FILE = 'golden_eval.csv'
FINAL_INTENTS = [
    'software_update',
    'battery_charging',
    'app_store',
    'account_apple_id',
    'connectivity',
    'keyboard_input',
    'device_issue',
    'payments_billing',
    'mail_messages',
    'media_accessories',
]


def clean_text(text):
    """Reuse the same light normalization pattern used by the workspace sampling/export scripts."""
    text = html.unescape(str(text))
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'&amp;', ' and ', text)
    # Keep question marks and apostrophes? They are not important for text features.
    text = re.sub(r'[^a-zA-Z0-9_\-\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_training_candidates(train_file=TRAIN_FILE):
    """Read weakly labeled training candidates from golden_candidates.csv.

    Important: The final golden_eval.csv is never used for training. It is used only for evaluation.
    """
    df = pd.read_csv(train_file, low_memory=False)
    if 'proposed_intent' not in df.columns:
        raise ValueError('golden_candidates.csv must contain proposed_intent for weak-label training.')
    if 'text' not in df.columns:
        raise ValueError('golden_candidates.csv must contain text for weak-label training.')
    df['text_clean'] = df['text'].fillna('').apply(clean_text)
    df = df[df['text_clean'].str.len() > 0]
    return df


def load_eval_set(eval_file=EVAL_FILE):
    """Load the final reviewed golden_eval.csv as the evaluation-only file."""
    df = pd.read_csv(eval_file, low_memory=False)
    if 'gold_intent' not in df.columns:
        raise ValueError('golden_eval.csv must contain gold_intent for evaluation.')
    if 'text' not in df.columns:
        raise ValueError('golden_eval.csv must contain text for evaluation.')

    # Keep the golden set separate from training. Remove ambiguous rows from the classifier evaluation
    # because ambiguous is review-only and not one of the final 10 intents.
    eval_df = df[df['gold_intent'] != 'ambiguous'].copy()
    eval_df['text_clean'] = eval_df['text'].fillna('').apply(clean_text)
    return eval_df


def train_majority(train_df):
    counts = train_df['proposed_intent'].value_counts()
    # majority class from training data
    majority = counts.idxmax()
    return majority


def evaluate_majority(train_label, eval_df):
    # Predict the training-derived majority for every evaluation example.
    y_true = eval_df['gold_intent']
    y_pred = [train_label] * len(eval_df)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=FINAL_INTENTS, average='macro', zero_division=0)
    per_f1 = {}
    for label in FINAL_INTENTS:
        per_f1[label] = f1_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)
    return acc, macro_f1, per_f1


def evaluate_lr(train_df, eval_df):
    # Train/evaluation split document:
    # 80/20 shuffled stratified split from golden_candidates.csv weak labels, random_state=42.
    # Only the train subset participates in fitting the vectorizer and classifier.
    # The final golden_eval.csv remains separate and is used only for evaluation.
    train_split, _ = train_test_split(
        train_df,
        train_size=0.8,
        test_size=0.2,
        shuffle=True,
        stratify=train_df['proposed_intent'],
        random_state=SEED,
    )

    X_train = train_split['text_clean'].tolist()
    y_train = train_split['proposed_intent'].tolist()

    # Use an ngram TF-IDF and deterministic LR.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        strip_accents='unicode',
        sublinear_tf=True,
    )
    Xtr = vectorizer.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        random_state=SEED,
        class_weight='balanced',
    )
    model.fit(Xtr, y_train)

    X_eval = vectorizer.transform(eval_df['text_clean'].tolist())
    y_pred = model.predict(X_eval)
    y_true = eval_df['gold_intent'].tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=FINAL_INTENTS, average='macro', zero_division=0)
    per_f1 = {}
    for label in FINAL_INTENTS:
        per_f1[label] = f1_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)

    return acc, macro_f1, per_f1, y_true, y_pred


def main():
    # Load separate data sources.
    train_df = load_training_candidates()
    eval_df = load_eval_set()

    # Majority-class baseline.
    majority_label = train_majority(train_df)
    acc_majority, macro_majority, f1_per_majority = evaluate_majority(majority_label, eval_df)

    # TF-IDF + LR baseline.
    acc_lr, macro_lr, f1_per_lr, y_true, y_pred = evaluate_lr(train_df, eval_df)

    # Write a readable report.
    lines = []
    lines.append('Intent Baseline Evaluation Report')
    lines.append('Source files:')
    lines.append(f'- Training source: {TRAIN_FILE} (weak proposed_intent labels only)')
    lines.append(f'- Evaluation source: {EVAL_FILE} (gold_intent labels only; ambiguous rows excluded from classifier evaluation)')
    lines.append('')
    lines.append('Training/evaluation split:')
    lines.append('1) Read weakly labeled training candidates from golden_candidates.csv.')
    lines.append('2) Use a deterministic shuffled stratified split of the weak training file: train_size=0.8, test_size=0.2, stratify=proposed_intent, random_state=42.')
    lines.append('3) Fit the TF-IDF vectorizer and Logistic Regression on the training split only.')
    lines.append('4) Evaluate on the final reviewed golden_eval.csv file only; do not use golden_eval.csv in any training step.')
    lines.append('')
    lines.append('Baseline results:')
    lines.append(f'Majority-class Accuracy: {acc_majority:.4f}')
    lines.append(f'Majority-class Macro F1: {macro_majority:.4f}')
    lines.append('Majority-class Per-intent F1:')
    for label in FINAL_INTENTS:
        lines.append(f'  {label}: {f1_per_majority[label]:.4f}')
    lines.append('')
    lines.append(f'TF-IDF + Logistic Regression Accuracy: {acc_lr:.4f}')
    lines.append(f'TF-IDF + Logistic Regression Macro F1: {macro_lr:.4f}')
    lines.append('TF-IDF + Logistic Regression Per-intent F1:')
    for label in FINAL_INTENTS:
        lines.append(f'  {label}: {f1_per_lr[label]:.4f}')

    # Print / save report.
    report_path = 'intent_baseline_results.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))

if __name__ == '__main__':
    main()
