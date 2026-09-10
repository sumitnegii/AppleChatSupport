#!/usr/bin/env python3
"""Grounded AppleSupport reply generation over retrieved historical conversations.

This layer stays strictly retrieval-grounded. It receives a customer message and
uses the existing historical retrieval implementation to retrieve top-K
conversation records from apple_support_conversation_index.csv.

The generator then writes a concise customer-support draft grounded in the
retrieved evidence. It is explicitly instructed not to invent unsupported
policies, actions, refunds, account changes, escalations, or any other
historical outcomes that are not supported by the retrieved conversations.

If an OpenAI/Azure OpenAI key is available in the environment, this file will
attempt to call a chat-completions model.
If not, it falls back to a deterministic template that is safe and grounded,
so the workspace remains reproducible without leaking credentials.
"""

import os
import re
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from historical_retrieval import cosine_search, clean_text

load_dotenv(dotenv_path='.env')

INDEX_CSV = 'apple_support_conversation_index.csv'
DEMO_OUTPUT = 'apple_support_grounded_reply_demo.txt'

# Keep the same model as the retrieval path unless the user sets another one.
OPENAI_MODEL = os.getenv('OPEN_AI_MODEL', os.getenv('OPENAI_MODEL', 'gpt-4o-mini'))
OPENROUTER_MODEL = os.getenv('OPEN_ROUTER_MODEL', 'openai/gpt-4o-mini')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')


def load_index(index_csv=INDEX_CSV) -> pd.DataFrame:
    """Load the saved historical conversation index from the retrieval artifact."""
    df = pd.read_csv(index_csv, low_memory=False)
    return df


def build_prompt(query: str, top: pd.DataFrame) -> str:
    """Create a grounded historical evidence prompt for the answerer model."""
    evidence_blocks = []
    for i, row in top.iterrows():
        conv_id = int(row['conversation_id'])
        score = float(row.get('sim_score', 0.0))
        evidence = {
            'conversation_id': conv_id,
            'retrieved_score': round(score, 4),
            'customer_turns': clean_text(row['customer_turns']),
            'support_turns': clean_text(row['support_turns']),
            'n_messages': int(row['n_messages']),
            'n_customer_turns': int(row['n_customer_turns']),
            'n_support_turns': int(row['n_support_turns']),
        }
        evidence_blocks.append(json.dumps(evidence, ensure_ascii=False))

    evidence_text = '\n\n'.join(
        [f'EVIDENCE #{i+1}: {block}' for i, block in enumerate(evidence_blocks)]
    )

    # Keep the directions explicit: the model receives historical evidence only.
    # It must not invent actions, policies, refunds, account changes, or escalations.
    prompt = f"""
You are writing a concise AppleSupport draft reply for a customer message.

Task:
Given the customer message and the retrieved historical AppleSupport conversation evidence,
produce a concise draft reply that is grounded only in the retrieved historical evidence.

Rules:
1. Use only the retrieved historical conversations as evidence.
2. Do not invent unsupported actions, policies, refunds, account changes, device swaps,
   escalations, or other outcomes not explicitly present in the historical evidence.
3. If the evidence is weak or ambiguous, say that the reply is a cautious historical template.
4. Keep the reply concise, supportive, and appropriate for a customer-support channel.
5. Include the retrieved conversation IDs/evidence used so the reply can be inspected.

Customer message:
{query}

Retrieved historical AppleSupport conversation evidence:
{evidence_text}

Output a concise draft reply for the customer and include a short "Evidence" list pointing at the retrieved conversation IDs used.
"""
    return prompt.strip()


def fallback_grounded_reply(query: str, top: pd.DataFrame) -> str:
    """Safe deterministic fallback when no model API is configured.

    This fallback prevents hallucinated support actions. It summarizes the evidence
    as historical support examples only and formulates a careful draft reply.
    """
    lines = []
    lines.append('Thanks for reaching out. Based on the historical AppleSupport conversation evidence, the closest matching cases suggest the following support path:')
    # Summarize evidence safely without inventing unsupported outcomes.
    for i, row in top.iterrows():
        conv_id = int(row['conversation_id'])
        support_text = clean_text(row['support_turns'])[:180]
        lines.append(f'- Conversation {conv_id}: {support_text.rstrip()}')
    lines.append('Please share the device model and current iOS version, then continue the conversation in DM as the historical examples do. Do not assume that a specific policy, refund, account change, or device swap is guaranteed from the historical evidence alone.')
    lines.append('Evidence IDs: ' + ', '.join([str(int(x)) for x in top['conversation_id'].tolist()]))
    return '\n'.join(lines)


def call_gemini_grounded_generation(prompt: str, model: str = GEMINI_MODEL,
                                    api_key: str = None) -> str:
    """Try a Gemini generation path from the secure .env key first."""
    api_key = api_key or os.getenv('GEMINI_API_KEY_MAIN')
    if not api_key:
        return ''

    try:
        # Use a lightweight HTTP request to Gemini REST endpoint.
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': 0.2, 'maxOutputTokens': 250},
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if 'candidates' in data and data['candidates']:
            return data['candidates'][0]['content']['parts'][0]['text'].strip()
        return ''
    except Exception as exc:
        print('Gemini generation failed:', type(exc).__name__, 'provider request error')
        return ''


def call_openai_grounded_generation(prompt: str, model: str = OPENAI_MODEL,
                                    api_key: str = None,
                                    base_url: str = None) -> str:
    """Call OpenAI using the secure .env-backed OPEN_AI_API_KEY_FALLBACK1 key if present."""
    api_key = api_key or os.getenv('OPEN_AI_API_KEY_FALLBACK1') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        return ''

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are writing a concise, grounded AppleSupport draft reply from historical evidence only.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.2,
            max_tokens=250,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print('OpenAI generation failed:', type(exc).__name__, 'provider request error')
        return ''


def call_openrouter_grounded_generation(prompt: str, model: str = OPENROUTER_MODEL,
                                        api_key: str = None,
                                        base_url: str = 'https://openrouter.ai/api/v1') -> str:
    """Call OpenRouter using the secure .env-backed OPEN_ROUTER_KEY_FALLBACK1 key if present."""
    api_key = api_key or os.getenv('OPEN_ROUTER_KEY_FALLBACK1')
    if not api_key:
        return ''

    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com',
            'X-Title': 'AppleSupport Grounded Retrieval Demo',
        }
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You are writing a concise, grounded AppleSupport draft reply from historical evidence only.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.2,
            'max_tokens': 250,
        }
        response = requests.post(base_url + '/chat/completions', headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as exc:
        print('OpenRouter generation failed:', type(exc).__name__, 'provider request error')
        return ''


def call_groq_grounded_generation(prompt: str, model: str = GROQ_MODEL,
                                  api_key: str = None,
                                  base_url: str = 'https://api.groq.com/openai/v1') -> str:
    """Call Groq via a Groq-compatible OpenAI API style endpoint using the secure .env GROQ_API_KEY key if present."""
    api_key = api_key or os.getenv('GROQ_API_KEY') or os.getenv('GROK_API_KEY_PRIMARY')
    if not api_key:
        return ''

    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You are writing a concise, grounded AppleSupport draft reply from historical evidence only.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.2,
            'max_tokens': 250,
        }
        response = requests.post(base_url + '/chat/completions', headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as exc:
        print('Groq generation failed:', type(exc).__name__, 'provider request error')
        return ''


def generate_grounded_reply(query: str,
                             index_csv: str = INDEX_CSV,
                             k: int = 3,
                             model: str = OPENAI_MODEL,
                             use_remote: bool = True) -> Dict[str, Any]:
    """Create one grounded draft reply from retrieved historical cases.

    Returns a dictionary with the customer query, the retrieved row records,
    the generated reply, and the evidence conversation IDs.
    """
    records = load_index(index_csv)
    top = cosine_search(query, records, k=k)

    prompt = build_prompt(query, top)

    # Requested provider order: Gemini -> OpenAI -> OpenRouter -> Groq/Grok -> deterministic fallback.
    generated = ''
    provider_used = 'deterministic_fallback'
    model_used = 'deterministic_template'
    remote_request_succeeded = False

    if use_remote:
        generated = call_gemini_grounded_generation(prompt, model=GEMINI_MODEL)
        if generated.strip():
            provider_used = 'gemini'
            model_used = GEMINI_MODEL
            remote_request_succeeded = True
        else:
            generated = call_openai_grounded_generation(prompt, model=model)
            if generated.strip():
                provider_used = 'openai'
                model_used = model
                remote_request_succeeded = True
            else:
                generated = call_openrouter_grounded_generation(prompt, model=OPENROUTER_MODEL)
                if generated.strip():
                    provider_used = 'openrouter'
                    model_used = OPENROUTER_MODEL
                    remote_request_succeeded = True
                else:
                    generated = call_groq_grounded_generation(prompt, model=GROQ_MODEL)
                    if generated.strip():
                        provider_used = 'groq'
                        model_used = GROQ_MODEL
                        remote_request_succeeded = True

    # Use deterministic fallback if no API key or failure occurs.
    if not generated.strip():
        generated = fallback_grounded_reply(query, top)
        provider_used = 'deterministic_fallback'
        model_used = 'deterministic_template'
        remote_request_succeeded = False

    # Build an evidence block for inspection.
    evidence_ids = [str(int(x)) for x in top['conversation_id'].tolist()]
    return {
        'query': query,
        'top_k': k,
        'evidence_ids': evidence_ids,
        'retrieved_score': top['sim_score'].tolist(),
        'retrieved_records': top,
        'draft_reply': generated,
        'provider_used': provider_used,
        'model_used': model_used,
        'remote_request_succeeded': remote_request_succeeded,
    }


def run_demo(query_list: List[str],
             index_csv: str = INDEX_CSV,
             k: int = 3,
             output_txt: str = DEMO_OUTPUT) -> str:
    """Run a small demo that shows query -> retrieval -> draft reply.

    The output file is a reproducible demonstration artifact.
    """
    lines = []
    lines.append('AppleSupport grounded reply-generation demo')
    lines.append('Retrieval source: ' + index_csv)
    lines.append('Grounded LLM path: Gemini -> OpenAI -> OpenRouter -> deterministic template fallback.')
    lines.append('No golden_eval.csv labels or taxonomy changes were used.')
    lines.append('')

    for i, query in enumerate(query_list, start=1):
        result = generate_grounded_reply(query, index_csv=index_csv, k=k)
        lines.append(f'Query {i}: {query}')
        lines.append('Evidence IDs: ' + ', '.join(result['evidence_ids']))
        lines.append('Retrieved sim scores: ' + ', '.join([f'{float(s):.4f}' for s in result['retrieved_score']]))
        lines.append('Customer evidence excerpt(s):')
        for j, row in result['retrieved_records'].iterrows():
            lines.append(f"  - conv_id={int(row['conversation_id'])}: {clean_text(row['customer_turns'])[:220]}")
        lines.append('Draft reply:')
        lines.append(result['draft_reply'])
        lines.append('')

    Path(output_txt).write_text('\n'.join(lines))
    return output_txt


def main():
    # Query list purely historical-similarity demonstration examples.
    demo_queries = [
        'battery drains after software update and phone won’t charge',
        'internet keeps dropping and wifi is unreliable',
        'keyboard is slow and autocorrect takes letters too late',
        'mail and messages are stopped syncing with iCloud',
    ]
    run_demo(demo_queries, output_txt=DEMO_OUTPUT)
    print(f'Wrote grounded reply demo to {DEMO_OUTPUT}')


if __name__ == '__main__':
    main()
