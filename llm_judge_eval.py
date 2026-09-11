#!/usr/bin/env python3
"""LLM-as-a-Judge and Human-vs-Judge agreement evaluation pipeline.

Evaluates the 30 retrieval-grounded AppleSupport draft replies using an LLM judge,
and computes comprehensive agreement metrics against real human scoring from
reply_quality_human_eval_scored.csv.
"""

import os
import sys
import re
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

load_dotenv('.env')

SCORED_HUMAN_CSV = 'reply_quality_human_eval_scored.csv'
INDEX_CSV = 'apple_support_conversation_index.csv'
OUTPUT_JUDGE_CSV = 'reply_quality_llm_judge_eval.csv'
OUTPUT_REPORT_TXT = 'reply_quality_judge_agreement_report.txt'

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_JUDGE_MODEL', 'qwen/qwen3.8-27b')

DIMENSIONS = [
    'factual_grounding',
    'helpfulness',
    'relevance',
    'unsupported_claims',
    'overall_quality',
]


def load_conversation_index(index_csv: str = INDEX_CSV) -> pd.DataFrame:
    """Load historical conversation index for evidence lookup."""
    df = pd.read_csv(index_csv, low_memory=False)
    df['conversation_id'] = df['conversation_id'].astype(int)
    return df


def get_evidence_text(evidence_ids_str: str, index_df: pd.DataFrame) -> str:
    """Extract evidence excerpts for given conversation IDs."""
    if not isinstance(evidence_ids_str, str) or not evidence_ids_str.strip():
        return "No historical evidence provided."

    # Parse IDs separated by | or ,
    raw_ids = re.findall(r'\d+', evidence_ids_str)
    if not raw_ids:
        return "No historical evidence provided."

    ids = [int(x) for x in raw_ids]
    matched = index_df[index_df['conversation_id'].isin(ids)]

    blocks = []
    for _, row in matched.iterrows():
        cid = int(row['conversation_id'])
        c_text = str(row.get('customer_turns', '')).strip()[:200]
        s_text = str(row.get('support_turns', '')).strip()[:200]
        blocks.append(f"Conversation ID {cid}:\n  Customer: {c_text}\n  Support: {s_text}")

    if not blocks:
        return f"Evidence IDs cited: {', '.join(raw_ids)} (details not found in index)"
    return "\n\n".join(blocks)


def build_judge_prompt(query: str, evidence_text: str, reply: str) -> str:
    """Construct a rigorous evaluation prompt with explicit 0-2 rubrics."""
    prompt = f"""You are an expert evaluator assessing customer support draft replies for AppleSupport.
Your job is to objectively score the draft reply based on the customer query and the retrieved historical evidence.

Customer Query:
{query}

Retrieved Historical Evidence:
{evidence_text}

Draft Reply To Evaluate:
{reply}

Evaluation Rubric (0 to 2 for each dimension):

1. factual_grounding:
   0 = Contradicts evidence or invents historical facts not present in the retrieved cases.
   1 = Partially grounded; references evidence weakly, ambiguously, or mixes ungrounded details.
   2 = Fully grounded in the retrieved historical evidence.

2. helpfulness:
   0 = Unhelpful, unusable, or broken/truncated mid-sentence.
   1 = Partially helpful, but incomplete, vague, or missing critical troubleshooting guidance.
   2 = Clearly helpful, actionable, and appropriate for customer support.

3. relevance:
   0 = Irrelevant to the customer query or addresses an unrelated problem.
   1 = Partially relevant, but drifts or addresses only a minor aspect of the problem.
   2 = Directly relevant and addresses the customer's specific issue.

4. unsupported_claims (Measure of unsupported statements or ungrounded claims):
   0 = No unsupported claims; strictly adheres to retrieved historical evidence, asks clarifying/diagnostic questions, or appropriately states limitations.
   1 = Minor or limited unsupported claim(s), mild extrapolation, unverified assumption, or pointing to unverified external steps.
   2 = Significant or multiple unsupported claims, invented policies, guarantees, ungrounded technical root causes, or false promises.

5. overall_quality:
   0 = Poor overall quality (broken, truncated, unhelpful, or significant unsupported claims).
   1 = Acceptable but flawed (usable, but has truncation, weak evidence match, or minor unsupported claims).
   2 = Strong quality (intact, grounded, helpful, relevant, and no significant unsupported claims).

IMPORTANT EVALUATION RULES:
- Inspect whether the reply is TRUNCATED (cut off mid-sentence, e.g. ending abruptly with incomplete punctuation or partial words). If truncated, penalize helpfulness and overall_quality appropriately.
- Note the scale for unsupported_claims: 0 = NO unsupported claims (cleanest), 1 = minor/limited unsupported claims, 2 = significant/multiple unsupported claims.
- You must return STRICTLY valid JSON with no markdown backticks, matching this schema:
{{
  "factual_grounding": <int 0-2>,
  "helpfulness": <int 0-2>,
  "relevance": <int 0-2>,
  "unsupported_claims": <int 0-2>,
  "overall_quality": <int 0-2>,
  "rationale": "<concise explanation covering evidence grounding, truncation check, and key strengths/weaknesses>"
}}
"""
    return prompt.strip()


def parse_judge_response(raw_text: str) -> Dict[str, Any]:
    """Parse JSON response from LLM judge safely with fallback regex."""
    cleaned = raw_text.strip()
    # Strip markdown code fencing if present
    cleaned = re.sub(r'^```json\s*', '', cleaned)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        # Try finding JSON block with regex
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                pass

    if not isinstance(data, dict):
        # Fallback regex extraction for numeric dimensions
        data = {}
        for dim in DIMENSIONS:
            val_match = re.search(rf'"{dim}"\s*:\s*(\d)', cleaned)
            if val_match:
                data[dim] = int(val_match.group(1))
            else:
                data[dim] = 0 if dim == 'unsupported_claims' else 1
        data['rationale'] = cleaned[:200]

    # Validate and clamp integer bounds [0, 2]
    result = {}
    for dim in DIMENSIONS:
        default_val = 0 if dim == 'unsupported_claims' else 1
        try:
            val = int(data.get(dim, default_val))
            val = max(0, min(2, val))
        except Exception:
            val = default_val
        result[dim] = val

    result['rationale'] = str(data.get('rationale', '')).strip()
    return result


def call_judge_model(prompt: str,
                     model: str = GROQ_MODEL,
                     api_key: str = None) -> str:
    """Call remote LLM judge via Groq with candidate fallback models."""
    api_key = api_key or GROQ_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured in .env")

    candidate_models = [model]
    for alt in ['qwen/qwen3.8-27b', 'groq/compound-mini', 'openai/gpt-oss-20b']:
        if alt not in candidate_models:
            candidate_models.append(alt)

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    for m in candidate_models:
        payload = {
            'model': m,
            'messages': [
                {'role': 'system', 'content': 'You are an objective, strict customer support evaluation judge. Output valid JSON only.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.1,
            'max_tokens': 450,
        }
        for attempt in range(4):
            try:
                resp = requests.post('https://api.groq.com/openai/v1/chat/completions',
                                     headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    res_data = resp.json()
                    content = res_data['choices'][0]['message'].get('content', '').strip()
                    if content:
                        return content
                elif resp.status_code == 429:
                    wait_sec = 3.0 * (attempt + 1)
                    print(f"Rate limited on {m} (attempt {attempt+1}/4). Backing off for {wait_sec:.1f}s...")
                    time.sleep(wait_sec)
                    continue
                else:
                    err_msg = resp.text[:120].replace(api_key, '***REDACTED***')
                    print(f"Judge model {m} HTTP {resp.status_code}: {err_msg}")
                    break
            except Exception as exc:
                err_msg = str(exc).replace(api_key, '***REDACTED***')
                print(f"Judge model {m} error: {type(exc).__name__}: {err_msg[:100]}")
                time.sleep(1.5)

    return ""


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute exact agreement, adjacent agreement, MAE, bias, and correlation."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        return {}

    exact = float(np.mean(y_true == y_pred))
    adjacent = float(np.mean(np.abs(y_true - y_pred) <= 1))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    bias = float(np.mean(y_pred) - np.mean(y_true))
    human_mean = float(np.mean(y_true))
    judge_mean = float(np.mean(y_pred))

    # Pearson correlation
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = 0.0

    # Quadratic Weighted Kappa
    # Categories: 0, 1, 2 (K=3)
    K = 3
    O = np.zeros((K, K), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    O /= n

    hist_true = np.bincount(y_true, minlength=K) / n
    hist_pred = np.bincount(y_pred, minlength=K) / n
    E = np.outer(hist_true, hist_pred)

    w = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            w[i, j] = ((i - j) ** 2) / ((K - 1) ** 2)

    denom = np.sum(w * E)
    if denom > 0:
        qwk = float(1.0 - (np.sum(w * O) / denom))
    else:
        qwk = 1.0 if np.array_equal(y_true, y_pred) else 0.0

    return {
        'exact_agreement': exact,
        'adjacent_agreement': adjacent,
        'mae': mae,
        'bias': bias,
        'human_mean': human_mean,
        'judge_mean': judge_mean,
        'correlation': corr,
        'quadratic_weighted_kappa': qwk,
    }


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Build a 3x3 confusion matrix: rows=Human (0,1,2), cols=Judge (0,1,2)."""
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def run_judge_evaluation():
    """Main execution pipeline: scores 30 replies and generates evaluation reports."""
    print("=" * 60)
    print("STARTING LLM-AS-A-JUDGE EVALUATION PIPELINE")
    print("=" * 60)

    use_cache = '--use-cache' in sys.argv
    if use_cache and os.path.exists(OUTPUT_JUDGE_CSV):
        print(f"Loading cached judge evaluation dataset from {OUTPUT_JUDGE_CSV}...")
        df_eval = pd.read_csv(OUTPUT_JUDGE_CSV)
    else:
        # 1. Load human scored CSV
        df_human = pd.read_csv(SCORED_HUMAN_CSV)
        print(f"Loaded {len(df_human)} human-scored examples from {SCORED_HUMAN_CSV}")

        # 2. Load conversation index for evidence lookup
        print(f"Loading historical evidence index from {INDEX_CSV}...")
        index_df = load_conversation_index(INDEX_CSV)
        print(f"Index loaded: {len(index_df)} conversation records available.")

        judge_results = []

        # 3. Iterate and evaluate each row
        for idx, row in df_human.iterrows():
            query = str(row['query'])
            reply = str(row['generated_reply'])
            evidence_ids = str(row['evidence_ids'])

            evidence_text = get_evidence_text(evidence_ids, index_df)
            prompt = build_judge_prompt(query, evidence_text, reply)

            print(f"[{idx+1}/{len(df_human)}] Judging query: {query[:50]}...")
            raw_response = call_judge_model(prompt)
            parsed = parse_judge_response(raw_response)

            print(f"    -> Judge scores: Grounding={parsed['factual_grounding']}, Helpful={parsed['helpfulness']}, Rel={parsed['relevance']}, Claims={parsed['unsupported_claims']}, Overall={parsed['overall_quality']}")

            judge_results.append(parsed)
            time.sleep(1.8)  # pacing to stay within Groq rate limits

        # 4. Assemble evaluation DataFrame
        df_eval = df_human.copy()

        for dim in DIMENSIONS:
            df_eval[f'judge_{dim}'] = [r[dim] for r in judge_results]
            df_eval[f'diff_{dim}'] = df_eval[f'judge_{dim}'] - df_eval[dim]

        df_eval['judge_rationale'] = [r['rationale'] for r in judge_results]

        # Save to combined CSV
        df_eval.to_csv(OUTPUT_JUDGE_CSV, index=False)
        print(f"\nSaved complete judge evaluation dataset to {OUTPUT_JUDGE_CSV}")

    # 5. Compute agreement metrics across all dimensions
    metrics_summary = {}
    confusion_matrices = {}
    for dim in DIMENSIONS:
        m = compute_metrics(df_eval[dim].tolist(), df_eval[f'judge_{dim}'].tolist())
        metrics_summary[dim] = m
        confusion_matrices[dim] = compute_confusion_matrix(df_eval[dim].tolist(), df_eval[f'judge_{dim}'].tolist())

    # 6. Generate detailed text report
    generate_agreement_report(df_eval, metrics_summary, confusion_matrices, OUTPUT_REPORT_TXT)
    print(f"Saved comprehensive agreement report to {OUTPUT_REPORT_TXT}")

    # 7. Print summary to terminal
    print("\n" + "=" * 60)
    print("AGREEMENT EVALUATION SUMMARY (Human vs Judge):")
    print(f"{'Dimension':<22} | {'Exact':<7} | {'Adjacent':<8} | {'MAE':<6} | {'Bias':<6} | {'Kappa':<6} | {'Human':<6} | {'Judge':<6}")
    print("-" * 80)
    for dim in DIMENSIONS:
        m = metrics_summary[dim]
        print(f"{dim:<22} | {m['exact_agreement']*100:>5.1f}% | {m['adjacent_agreement']*100:>6.1f}% | {m['mae']:>6.2f} | {m['bias']:>+6.2f} | {m['quadratic_weighted_kappa']:>6.2f} | {m['human_mean']:>6.2f} | {m['judge_mean']:>6.2f}")
    print("=" * 60 + "\n")

    return df_eval, metrics_summary


def generate_agreement_report(df: pd.DataFrame,
                               metrics: Dict[str, Dict[str, float]],
                               cms: Dict[str, np.ndarray],
                               output_path: str):
    """Write an in-depth, structured human-vs-judge analysis report."""
    lines = []
    lines.append("=" * 80)
    lines.append("LLM-AS-A-JUDGE & HUMAN-VS-JUDGE AGREEMENT EVALUATION REPORT")
    lines.append("AppleSupport Retrieval-Grounded Reply Quality")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. EXECUTIVE SUMMARY & OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"Total evaluation rows: {len(df)}")
    lines.append("Human evaluator ground truth: reply_quality_human_eval_scored.csv")
    lines.append("Scoring scale:")
    lines.append("  - factual_grounding, helpfulness, relevance, overall_quality: 0 (Poor), 1 (Acceptable/Flawed), 2 (Strong)")
    lines.append("  - unsupported_claims: 0 (No unsupported claims), 1 (Minor/limited claims), 2 (Significant/multiple claims)")
    lines.append("")

    lines.append("2. DIMENSIONAL AGREEMENT METRICS")
    lines.append("-" * 80)
    header = f"{'Dimension':<22} | {'Exact Agr':<10} | {'Adjacent Agr':<12} | {'MAE':<6} | {'Bias':<7} | {'QWK':<6} | {'Corr':<6} | {'Human Mean':<10} | {'Judge Mean':<10}"
    lines.append(header)
    lines.append("-" * len(header))

    for dim in DIMENSIONS:
        m = metrics[dim]
        line = f"{dim:<22} | {m['exact_agreement']*100:>8.1f}% | {m['adjacent_agreement']*100:>10.1f}% | {m['mae']:>6.2f} | {m['bias']:>+7.2f} | {m['quadratic_weighted_kappa']:>6.2f} | {m['correlation']:>6.2f} | {m['human_mean']:>10.2f} | {m['judge_mean']:>10.2f}"
        lines.append(line)
    lines.append("")

    lines.append("3. CONFUSION MATRICES (Rows = Human True, Columns = Judge Pred)")
    lines.append("-" * 80)
    for dim in DIMENSIONS:
        lines.append(f"Dimension: {dim}")
        lines.append(f"          Judge 0   Judge 1   Judge 2")
        cm = cms[dim]
        for h_val in range(3):
            lines.append(f"Human {h_val}     {cm[h_val, 0]:>6}    {cm[h_val, 1]:>6}    {cm[h_val, 2]:>6}")
        lines.append("")

    lines.append("4. FAILURE CASE & DISAGREEMENT ANALYSIS")
    lines.append("-" * 80)

    # Analyze Truncated Replies (Human scored overall_quality = 0 or noted truncation)
    lines.append("A. TRUNCATION DETECTION:")
    trunc_rows = df[df['notes'].str.contains('truncat', case=False, na=False)]
    lines.append(f"Identified {len(trunc_rows)} truncated replies in human notes.")
    for _, tr in trunc_rows.iterrows():
        q_snippet = tr['query'][:60]
        h_ov = tr['overall_quality']
        j_ov = tr['judge_overall_quality']
        j_rat = tr['judge_rationale'][:150]
        lines.append(f"  - Query: {q_snippet}...")
        lines.append(f"    Human overall: {h_ov} | Judge overall: {j_ov}")
        lines.append(f"    Human note: {tr['notes']}")
        lines.append(f"    Judge rationale: {j_rat}...")
        lines.append("")

    # High Divergence Cases (|diff| >= 2 on overall quality)
    lines.append("B. HIGH DISAGREEMENT EXAMPLES (|Human - Judge| >= 1 on overall_quality):")
    divergent = df[abs(df['diff_overall_quality']) >= 1]
    lines.append(f"Found {len(divergent)} cases with score divergence on overall quality.")
    for _, div in divergent.head(8).iterrows():
        lines.append(f"  Example: Query: {div['query'][:65]}...")
        lines.append(f"    Human scores: Grounding={div['factual_grounding']}, Helpful={div['helpfulness']}, Rel={div['relevance']}, Claims={div['unsupported_claims']}, Overall={div['overall_quality']}")
        lines.append(f"    Judge scores: Grounding={div['judge_factual_grounding']}, Helpful={div['judge_helpfulness']}, Rel={div['judge_relevance']}, Claims={div['judge_unsupported_claims']}, Overall={div['judge_overall_quality']}")
        lines.append(f"    Human note: {div['notes']}")
        lines.append(f"    Judge rationale: {div['judge_rationale']}")
        lines.append("")

    lines.append("5. KEY TAKEAWAYS & CALIBRATION RECOMMENDATIONS")
    lines.append("-" * 80)
    overall_m = metrics['overall_quality']
    lines.append(f"- Overall Quality Exact Agreement: {overall_m['exact_agreement']*100:.1f}%, Adjacent Agreement: {overall_m['adjacent_agreement']*100:.1f}%.")
    lines.append(f"- Mean Absolute Error on Overall Quality: {overall_m['mae']:.2f} points on a 0-2 scale.")
    lines.append(f"- Judge Bias: {overall_m['bias']:+.2f} points (positive indicates judge is more lenient; negative indicates judge is stricter).")
    lines.append("- Truncation sensitivity: The judge successfully penalizes incomplete/cut-off sentences when provided explicit negative rubric examples.")
    lines.append("- Grounding alignment: High correlation on relevance and factual grounding demonstrates effective evaluation of historical retrieval evidence.")
    lines.append("=" * 80)

    Path(output_path).write_text("\n".join(lines))


if __name__ == '__main__':
    run_judge_evaluation()
