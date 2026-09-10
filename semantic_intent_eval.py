import pandas as pd
import numpy as np
import re
import html
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

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
    """Reuse the same light normalization as baseline_intent_eval.py."""
    text = html.unescape(str(text))
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'&amp;', ' and ', text)
    text = re.sub(r'[^a-zA-Z0-9_\-\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_training_candidates(train_file=TRAIN_FILE):
    """Load weakly labeled candidates; no final evaluation labels enter here."""
    df = pd.read_csv(train_file, low_memory=False)
    if 'proposed_intent' not in df.columns:
        raise ValueError('golden_candidates.csv must contain proposed_intent for weak-label training.')
    if 'text' not in df.columns:
        raise ValueError('golden_candidates.csv must contain text for weak-label training.')
    df['text_clean'] = df['text'].fillna('').apply(clean_text)
    df = df[df['text_clean'].str.len() > 0]
    return df


def load_eval_set(eval_file=EVAL_FILE):
    """Load final reviewed golden_eval.csv as evaluation-only and remove ambiguous rows."""
    df = pd.read_csv(eval_file, low_memory=False)
    if 'gold_intent' not in df.columns:
        raise ValueError('golden_eval.csv must contain gold_intent for evaluation.')
    if 'text' not in df.columns:
        raise ValueError('golden_eval.csv must contain text for evaluation.')
    eval_df = df[df['gold_intent'] != 'ambiguous'].copy()
    eval_df['text_clean'] = eval_df['text'].fillna('').apply(clean_text)
    return eval_df


def train_majority(train_df):
    counts = train_df['proposed_intent'].value_counts()
    return counts.idxmax()


def evaluate_majority(train_label, eval_df):
    y_true = eval_df['gold_intent'].tolist()
    y_pred = [train_label] * len(eval_df)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=FINAL_INTENTS, average='macro', zero_division=0)
    per_f1 = {}
    for label in FINAL_INTENTS:
        per_f1[label] = f1_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)
    return acc, macro_f1, per_f1


def evaluate_lr(train_df, eval_df):
    # Same deterministic split convention as baseline script.
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
    return acc, macro_f1, per_f1


def evaluate_sentence_embedding_lr(train_df, eval_df):
    # Same split convention as baseline script: deterministic shuffled stratified 80/20 on weak labels.
    train_split, _ = train_test_split(
        train_df,
        train_size=0.8,
        test_size=0.2,
        shuffle=True,
        stratify=train_df['proposed_intent'],
        random_state=SEED,
    )

    # Lightweight SentenceTransformer model; deterministic inference by model.
    # Embedding model is purposely simple and explainable.
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    X_train = embed_model.encode(train_split['text_clean'].tolist(), show_progress_bar=False)
    y_train = train_split['proposed_intent'].tolist()
    X_eval = embed_model.encode(eval_df['text_clean'].tolist(), show_progress_bar=False)
    y_true = eval_df['gold_intent'].tolist()

    # Simple classifier on the embeddings.
    model = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        random_state=SEED,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=FINAL_INTENTS, average='macro', zero_division=0)
    per_f1 = {}
    for label in FINAL_INTENTS:
        per_f1[label] = f1_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)
    return acc, macro_f1, per_f1


def main():
    train_df = load_training_candidates()
    eval_df = load_eval_set()

    # Baseline 1: majority class.
    majority_label = train_majority(train_df)
    acc_maj, f1_macro_maj, per_f1_maj = evaluate_majority(majority_label, eval_df)

    # Baseline 2: TF-IDF + LR.
    acc_tfidf_lr, f1_macro_tfidf_lr, per_f1_tfidf_lr = evaluate_lr(train_df, eval_df)

    # New semantic embedding + logistic regression comparator.
    acc_st_lr, f1_macro_st_lr, per_f1_st_lr = evaluate_sentence_embedding_lr(train_df, eval_df)

    lines = []
    lines.append('Intent Semantic Baseline Comparison Report')
    lines.append('Source files:')
    lines.append(f'- Training source: {TRAIN_FILE} (weak proposed_intent labels only)')
    lines.append(f'- Evaluation source: {EVAL_FILE} (gold_intent labels only; ambiguous rows excluded from classifier evaluation)')
    lines.append('')
    lines.append('Shared preprocessing/evaluation convention:')
    lines.append('1) Reuse clean_text normalization from baseline_intent_eval.py.')
    lines.append('2) Keep the weak-label training file separate from the final reviewed golden_eval.csv.')
    lines.append('3) Use the same deterministic shuffled stratified split: train_size=0.8, test_size=0.2, stratify=proposed_intent, random_state=42.')
    lines.append('4) Evaluate on the final reviewed golden_eval.csv file only; ambiguous rows must be excluded from classifier evaluation.')
    lines.append('')
    lines.append('Results:')
    lines.append(f'Majority Accuracy: {acc_maj:.4f}')
    lines.append(f'Majority Macro F1: {f1_macro_maj:.4f}')
    lines.append('Majority Per-intent F1:')
    for label in FINAL_INTENTS:
        lines.append(f'  {label}: {per_f1_maj[label]:.4f}')
    lines.append('')
    lines.append(f'TF-IDF + Logistic Regression Accuracy: {acc_tfidf_lr:.4f}')
    lines.append(f'TF-IDF + Logistic Regression Macro F1: {f1_macro_tfidf_lr:.4f}')
    lines.append('TF-IDF + Logistic Regression Per-intent F1:')
    for label in FINAL_INTENTS:
        lines.append(f'  {label}: {per_f1_tfidf_lr[label]:.4f}')
    lines.append('')
    lines.append(f'SentenceTransformer Embedding + Logistic Regression Accuracy: {acc_st_lr:.4f}')
    lines.append(f'SentenceTransformer Embedding + Logistic Regression Macro F1: {f1_macro_st_lr:.4f}')
    lines.append('SentenceTransformer Embedding + Logistic Regression Per-intent F1:')
    for label in FINAL_INTENTS:
        lines.append(f'  {label}: {per_f1_st_lr[label]:.4f}')

    # Write report.
    with open('intent_semantic_comparison_results.txt', 'w') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))

if __name__ == '__main__':
    main()
