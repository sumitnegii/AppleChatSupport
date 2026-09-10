import argparse
import pandas as pd
import csv
import os

CANDIDATE_FILE = 'golden_candidates.csv'
OUTPUT_FILE = 'golden_eval.csv'

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


def init_eval_file(candidate_file=CANDIDATE_FILE, output_file=OUTPUT_FILE):
    """Create golden_eval.csv from the candidate file.

    The original candidate columns remain unchanged, and new review columns are added:
    gold_intent, label_confidence, notes, ambiguous.
    """
    df = pd.read_csv(candidate_file, low_memory=False)
    # Ensure original columns remain untouched.
    required_cols = [
        'tweet_id',
        'conversation_id',
        'text',
        'proposed_intent',
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in {candidate_file}: {missing}')

    # Add evaluation columns, but do NOT auto-copy proposed_intent into gold_intent.
    df['gold_intent'] = ''
    df['label_confidence'] = ''
    df['notes'] = ''
    df['ambiguous'] = False

    # Keep requested report columns and output order stable.
    output_columns = [
        'tweet_id',
        'conversation_id',
        'text',
        'proposed_intent',
        'gold_intent',
        'label_confidence',
        'notes',
        'ambiguous',
    ]

    out = df[output_columns]
    out.to_csv(output_file, index=False)
    print(f'Created {output_file} with {len(out)} rows.')


def prompt_for_label(idx, row):
    print('\n' + '=' * 80)
    print(f'Example {idx + 1} / 250')
    print('=' * 80)
    print(f"tweet_id: {row['tweet_id']}")
    print(f"conversation_id: {row['conversation_id']}")
    print(f"proposed_intent: {row['proposed_intent']}")
    print('text:')
    print(row['text'])
    print()

    allowed = '/'.join(FINAL_INTENTS)
    print('Allowed intents:', allowed)
    print('Enter one of the 10 final intents exactly, or type ambiguous.')
    label = input('gold_intent: ').strip()
    if label.lower() == 'ambiguous':
        return 'ambiguous', True

    if label not in FINAL_INTENTS:
        print('Invalid intent; please repeat using one of the allowed labels.')
        return prompt_for_label(idx, row)

    return label, False


def review_eval(input_file=OUTPUT_FILE, output_file=OUTPUT_FILE):
    """Interactive CLI loop that handles manual review of all examples.

    On each loop the user provides:
    - gold_intent (one of the permitted 10 intents or ambiguous)
    - label_confidence (1-5)
    - notes (optional free text)
    - ambiguous flag (True/False)

    The row's proposed_intent is never copied automatically into gold_intent.
    """
    if not os.path.exists(input_file):
        init_eval_file()

    df = pd.read_csv(input_file, low_memory=False)
    # Ensure required extra columns exist.
    for c in ['gold_intent', 'label_confidence', 'notes', 'ambiguous']:
        if c not in df.columns:
            df[c] = ''
    df['ambiguous'] = df['ambiguous'].fillna(False)

    # Shallow validate shape.
    if len(df) != 250:
        print(f'Found {len(df)} rows; expected 250 rows in {input_file}.')

    for idx, row in df.iterrows():
        # Skip completed examples.
        if pd.notna(row['gold_intent']) and str(row['gold_intent']).strip() and str(row['gold_intent']).strip() != 'ambiguous':
            # Already labeled; still allow edits but don't force stop.
            print(f'Skipping already labeled example {idx+1}')
            continue

        label, is_ambiguous = prompt_for_label(idx, row)

        # Confidence prompt.
        print('Confidence from 1-5: 1=low, 5=high')
        confidence = input('label_confidence: ').strip()
        try:
            c = int(confidence)
            if c < 1 or c > 5:
                raise ValueError
        except Exception:
            print('Please enter a whole number from 1 to 5.')
            confidence = input('label_confidence: ').strip()
            try:
                c = int(confidence)
            except Exception:
                c = 3

        # Optional notes.
        notes = input('notes (optional): ').strip()

        # Assign sentinel values.
        df.at[idx, 'gold_intent'] = label if not is_ambiguous else 'ambiguous'
        df.at[idx, 'label_confidence'] = str(c)
        df.at[idx, 'notes'] = notes
        df.at[idx, 'ambiguous'] = bool(is_ambiguous)

        # Persist after each example.
        df.to_csv(output_file, index=False)
        print('Saved row', idx + 1)

    print('\nManual review complete. Saved', output_file)


def main():
    parser = argparse.ArgumentParser(description='Create and/or review golden_eval.csv for manual labeling.')
    parser.add_argument('--init', action='store_true', help='Create the initial golden_eval.csv scaffold from golden_candidates.csv.')
    parser.add_argument('--review', action='store_true', help='Launch the interactive manual labeling review loop.')
    args = parser.parse_args()

    if args.init:
        init_eval_file()
    elif args.review:
        review_eval()
    else:
        # Default: create the scaffold, then give a quick command hint.
        init_eval_file()
        print('Run: python3 manual_label_workflow.py --review to label rows manually.')

if __name__ == '__main__':
    main()
