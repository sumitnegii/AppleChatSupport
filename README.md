# AppleSupport AI Agent

An automated customer support agent for Apple customer inquiries on Twitter. The system classifies incoming customer messages into a 10-class intent taxonomy, retrieves relevant historical AppleSupport conversations from a local FAISS vector index, drafts a reply grounded strictly in the retrieved evidence, and decides whether to auto-handle the query or escalate to a human agent with an explicit reason.

---

## 1. Problem

### What the agent does
The agent handles incoming customer support messages on Apple products and services. For each query, it:
1. Classifies the issue into a 10-class intent taxonomy.
2. Searches historical AppleSupport conversations for relevant precedent.
3. Checks whether the retrieved evidence is close enough to answer safely.
4. Drafts a concise response grounded only in the retrieved conversations.
5. Evaluates operational risk and decides whether to auto-handle or escalate to a human agent.

### What "good" means
- **Accurate intent classification**: The system routes the query to the correct operational domain without guessing on underspecified messages.
- **Strict historical grounding**: The draft reply must only recommend steps verified in past AppleSupport conversations. It must never invent Apple policies, warranty terms, replacement offers, or release dates.
- **Conservative escalation**: High-risk issues (billing disputes, Apple ID lockouts, hardware hazards like swollen batteries) must be escalated to a human agent 100% of the time.
- **Explainability**: Every escalation decision has a human-readable reason and lists the specific risk triggers that fired.

### What we deliberately did not build
- **An ungrounded chatbot**: We do not let an LLM answer from general pretraining knowledge without historical evidence.
- **An "escalate" intent class**: Escalation is a risk policy, not an issue topic. A battery question could be a routine tip request (auto-handle) or a burning battery (immediate human safety escalation).
- **A hosted vector database**: We avoided Pinecone, Weaviate, or cloud vector infrastructure to keep the system lightweight, reproducible, and runnable locally in minutes.
- **Uncalibrated black-box routing**: Escalation decisions are based on explicit, inspectable safety rules and threshold checks, not opaque probability cutoffs.

---

## 2. Data and intents

### Dataset selection
We used the Kaggle "Customer Support on Twitter" dataset (~2.8M tweets). We focused on `@AppleSupport` because it is the largest and cleanest single brand in the dataset, with multi-turn conversations covering both hardware and software issues.

### Conversation reconstruction
Raw tweets were grouped by `conversation_id` and ordered by `created_at` / `thread_order`. This produced 80,717 reconstructed multi-turn conversations (204,238 total tweets, 97,370 inbound customer tweets).

### 10-Intent taxonomy
We clustered recurring customer issues into 10 operational intents:
1. `software_update`: iOS upgrade errors, update freezes, installation issues.
2. `battery_charging`: Battery drain, device overheating, charging cable problems.
3. `app_store`: App download failures, updates, purchases, App Store errors.
4. `account_apple_id`: Apple ID lockouts, 2FA, password reset, iCloud login.
5. `connectivity`: Wi-Fi dropping, Bluetooth pairing, cellular data issues.
6. `keyboard_input`: Predictive text glitches (e.g. the iOS 11 "I" bug), typing lag.
7. `device_issue`: Screen freezing, unexpected restarts, touch responsiveness.
8. `payments_billing`: Unrecognized charges, double billing, refund inquiries.
9. `mail_messages`: iMessage activation, SMS delivery, Mail app syncing.
10. `media_accessories`: AirPods pairing, Apple Music playback, camera, headphones.

An explicit `ambiguous` label is used for queries that are too short or underspecified to classify reliably.

### Golden evaluation set
We built a 250-example golden set (`golden_eval.csv`):
- **244 rows** have clear single intents.
- **6 rows** are ambiguous queries (e.g. "please help me", "it stopped working").
- Examples were sampled across intents to ensure representation of both frequent categories (updates, battery) and rarer ones (billing, accessories). Each row was manually reviewed with confidence ratings and review notes.

---

## 3. Approach

The pipeline processes messages through six sequential stages:

```
Customer message
  → [1] Normalization: Clean text and validate token length
  → [2] Intent classification: TF-IDF + Logistic Regression (10 classes)
  → [3] Historical retrieval: Local FAISS IndexFlatIP (top-3 cases, 2,000 vectors)
  → [4] Evidence check: Validate top similarity score >= 0.28
  → [5] Grounded reply: LLM generation conditioned strictly on retrieved evidence
  → [6] Escalation decision: Rule-based safety check with explicit reason
  → AUTO-HANDLE or HUMAN ESCALATION
```

### Key components
- **Classifier**: TF-IDF (1-2 n-grams) with a balanced Logistic Regression model. A SentenceTransformer (`all-MiniLM-L6-v2`) classifier is also evaluated as a semantic comparator.
- **Low-confidence safeguard**: If the top intent probability is below 0.13 (just above the 10% uniform prior for 10 classes), the query is flagged for human review. Note: this probability is uncalibrated; the threshold is an engineering safeguard.
- **Retrieval layer**: 2,000 multi-turn AppleSupport conversations encoded with `all-MiniLM-L6-v2` (384 dimensions) and indexed using FAISS `IndexFlatIP`. Vectors are L2-normalized, making inner product search mathematically identical to cosine similarity.
- **Grounded generator**: Passes the customer query and top-3 historical conversations to an LLM with strict instructions not to invent policies, refunds, or device swaps. If no API key is set, a deterministic template formats the evidence safely.
- **Escalation rules**:
  1. *Explicit human request*: "speak to a person", "human agent".
  2. *Account security*: locked out, hacked, Apple ID disabled, 2FA reset.
  3. *Financial dispute*: unrecognized charge, double charge, refund request.
  4. *Hardware safety / data loss*: swollen battery, smoking device, lost all data.
  5. *Insufficient retrieval score*: top similarity < 0.28 (out-of-domain issue).
  6. *Low intent confidence*: classifier probability < 0.13.
  7. *Unsupported draft commitments*: draft contains unverified promises ("we will refund").

---

## 4. Evaluation

### A. Intent classification baselines

We evaluated models against the 244 non-ambiguous golden examples under three conditions:
1. **Headline condition**: Trained on weakly labeled candidate data (`golden_candidates.csv`). Note: this split had 80.3% text overlap with the evaluation set (see Section 5).
2. **Strict out-of-sample holdout (N=48)**: Only evaluation examples whose text never appeared in the weak training set. 0% leakage.
3. **5-fold stratified cross-validation on golden labels (N=244)**: Evaluated directly on the human labels with 0% leakage across folds.

| Model | Evaluation Condition | Accuracy | Macro F1 | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Majority baseline** | Always predicts `software_update` | 0.0943 | 0.0172 | Trivial baseline (most common training class) |
| **TF-IDF + Logistic Regression** | Headline (weak train split) | 0.6844 | 0.6681 | 80.3% text overlap with train set |
| **Semantic (MiniLM) + LR** | Headline (weak train split) | 0.7008 | 0.6901 | 80.3% text overlap with train set |
| **TF-IDF + Logistic Regression** | **Strict holdout (N=48)** | **0.5625** | **0.5201** | **0% leakage, unseen test split** |
| **Semantic (MiniLM) + LR** | **Strict holdout (N=48)** | **0.6250** | **0.6114** | **0% leakage, unseen test split** |
| **TF-IDF + Logistic Regression** | **5-fold CV on human labels** | **0.8281** | **0.7978** | **0% leakage, trained on human labels (±0.027)** |
| **Semantic (MiniLM) + LR** | **5-fold CV on human labels** | **0.7991** | **0.7851** | **0% leakage, trained on human labels (±0.060)** |

**Understanding the gap**: When trained on noisy heuristic pseudo-labels, out-of-sample accuracy on unseen text is 56.3%–62.5%. When trained and evaluated directly on clean human labels via 5-fold CV, TF-IDF reaches 82.8% accuracy. This 82.8% is useful context, but it is based on a small 244-example dataset and should not be treated as a large-scale benchmark.

### B. Historical vector retrieval
- **Index**: Local FAISS `IndexFlatIP` storing 2,000 multi-turn conversation vectors (384 dimensions).
- **Parity verification**: FAISS search matches exact brute-force cosine similarity 100.0% of the time (identical conversation IDs and ranking).
- **Self-retrieval sanity check (N=25 sample)**:
  - Top-1 hit rate: **1.0000**
  - MRR@5: **1.0000**
  *Note: This is a sanity check verifying that a conversation's customer turns can rediscover their own full transcript. It is not an independent human relevance evaluation.*

### C. Reply quality: Human evaluation vs. LLM-as-a-judge (N=30)
30 replies generated from real historical cases were evaluated across 5 dimensions on a 0–2 scale by a human evaluator, then scored by an LLM judge using the identical rubric:

| Dimension | Human Mean | Judge Mean | Exact Agr | Adjacent Agr | MAE | Bias | QWK |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`overall_quality`** | 1.33 | 1.20 | **60.0%** | **100.0%** | **0.40** | -0.13 | **0.57** |
| **`helpfulness`** | 1.37 | 1.23 | **60.0%** | **100.0%** | **0.40** | -0.13 | **0.55** |
| **`relevance`** | 1.63 | 1.93 | **70.0%** | **100.0%** | **0.30** | +0.30 | **0.22** |
| **`factual_grounding`** | 1.67 | 1.80 | **60.0%** | **100.0%** | **0.40** | +0.13 | **0.14** |
| **`unsupported_claims`** | 0.63 | 0.13 | **60.0%** | **90.0%** | **0.50** | -0.50 | **0.25** |

- On `overall_quality`, the judge achieved **60% exact agreement** and **100% adjacent agreement** (within 1 point), with a Quadratic Weighted Kappa of 0.57.
- On `unsupported_claims`, the judge correctly agreed on 15/15 clean replies (100% on zero-claim cases), but was more lenient than the human on minor claims (judge mean 0.13 vs. human mean 0.63).
- **Important**: The LLM judge is an automated evaluator, not ground truth. It has a slight leniency bias on factual claims.

### D. Escalation behavior
Tested across 11 representative queries spanning routine issues, edge cases, and safety hazards (`apple_support_escalation_demo.txt`):
- **Auto-handled (45.5%)**: Routine troubleshooting queries with clear intent and high retrieval similarity (>= 0.28).
- **Escalated (54.5%)**: Every safety hazard was caught:
  - Account lockout ("disabled account, locked out") -> Escalated (`account_security_access`).
  - Double charge ("charged twice for subscription") -> Escalated (`payment_refund_dispute`).
  - Swollen battery ("phone burning up, battery swollen") -> Escalated (`severe_hardware_safety_data_loss`).
  - Explicit human request ("want to speak with a human agent") -> Escalated (`explicit_human_request`).
  - Out-of-domain query ("install Ubuntu on Apple Watch", similarity 0.19) -> Escalated (`insufficient_retrieval_grounding`).
  *Note: The 45.5% / 54.5% split is an illustration of safety rule behavior on a curated test set, not a production traffic distribution.*

---

## 5. What is misleading about my headline number?

To be transparent about the limits of our results:

1. **Classifier text leakage in the original headline**: The initial evaluation script (`baseline_intent_eval.py`) trained on an 80% split of `golden_candidates.csv` and evaluated on `golden_eval.csv`. Because both files originated from the same candidate pool, **196 of the 244 evaluation rows (80.3%) had their exact text seen during training** (with weak labels during training and gold labels during evaluation). The original 68.4% accuracy is therefore inflated by text memorization. On the clean 48-row unseen holdout, accuracy is **56.3%** (TF-IDF) and **62.5%** (Semantic).
2. **Small golden set (250 examples, 244 non-ambiguous)**: While 250 rows meets the assignment requirement, 10 classes mean rarer categories have very few examples (`media_accessories` has 7, `app_store` has 10). In a 7-example class, a single misclassification shifts class recall by 14.3%.
3. **5-fold CV is not an independent test set**: The 82.8% CV result proves that TF-IDF learns well from clean human labels, but it was trained and evaluated on the same 244 examples. It should not be cited as an external benchmark.
4. **Reply quality evaluation has only 30 examples**: Human review of 30 generated replies showed 60% exact agreement with the LLM judge. With N=30, the standard error is ~8.9%. This confirms rough alignment, but is not large enough to catch rare hallucinations.
5. **Self-retrieval MRR@5 is a sanity check, not gold relevance**: The self-retrieval check (1.0000 on N=25 sample) measures whether the model finds a conversation from its own text. It does not prove that the retrieved conversation offers the best possible troubleshooting advice for a new user.
6. **2,000-conversation subset**: We indexed 2,000 out of ~80,000 reconstructed conversations. While this subset covers common issues well, niche hardware or software bugs may have no matching precedent in the index.
7. **Twitter dataset quirks**: Twitter support replies are notoriously terse and frequently say "DM us your serial number". The generator sometimes mimics this brevity rather than providing a complete self-contained answer.
8. **LLM judge is not ground truth**: The judge agreed with humans 100% within 1 point, but showed a measurable leniency bias (+0.13 on factual grounding, -0.50 on unsupported claims). It cannot replace human spot-checks.

---

## 6. Failure analysis

Five real failure cases from our evaluation logs (`reply_quality_human_eval_scored.csv` and `apple_support_escalation_demo.txt`):

### 1. Mid-sentence generation truncation (Row 26)
- **Customer**: *"Dog it’s hard to remember to tap something else AFTER u tap “I”. Fuckin up ALL my tweets! @AppleSupport Get this box A shit together! I shouldn’t have to make an ALTERNATIVE TEXT..."*
- **Generated reply**: `'Hi there,\n\nIt sounds like the “I” key is behaving unexpectedly on your keyboard.\nTo'`
- **What went wrong**: The generation cut off mid-word after two lines. Scored 0 on helpfulness and overall quality.
- **Hypothesis**: Output token budget or generation buffer cutoff severed the response.
- **Possible fix**: Add a post-generation validation check that rejects any response not ending in complete terminal punctuation (`.`, `!`, `?`) and triggers a regeneration.

### 2. Unauthorized organizational commitments (Row 9)
- **Customer**: *"@AppleSupport It would be great if you guys introduce gift cards or other payment methods like online banking or through third party apps like Paytm..."*
- **Generated reply**: *"...While we don’t have immediate details on upcoming payment options, we will forward your comments to our product team..."*
- **What went wrong**: Scored 2 on unsupported claims. Tier-1 automated support cannot make commitments on behalf of product teams.
- **Hypothesis**: The LLM defaulted to conversational customer-service clichés learned during RLHF.
- **Possible fix**: Add negative prompt constraints explicitly forbidding phrases like "we will forward" or "our team is working on", and provide standard feedback links instead.

### 3. Re-recommending steps the customer already tried (Row 14)
- **Customer**: *"Why is predictive text changing the letter “I” to I️? I️ reset my keyboard, turned my phone on/off, and it’s still doing it."*
- **Generated reply**: Advised resetting the keyboard dictionary and restarting the device.
- **What went wrong**: Scored 1 on helpfulness for telling the user to do what they just said they already did.
- **Hypothesis**: The retrieval matched cases where the standard fix was resetting the keyboard dictionary, and the generator ignored the customer's negation context.
- **Possible fix**: Add a pre-processing step that detects phrases like "already tried", "did reset", or "restarted" and passes them as negative constraints into the generation prompt.

### 4. Parametric knowledge leakage (Row 1)
- **Customer**: *"@AppleSupport after update of ios 11.0.2 my iphone 6s screen freeze and phone has become slow with battery drain . Please help"*
- **Generated reply**: Claimed that *"iOS 11.1 resolved these issues"*.
- **What went wrong**: Scored 1 on unsupported claims. None of the retrieved historical cases mentioned that iOS 11.1 fixed this problem.
- **Hypothesis**: The model leaked parametric knowledge about historical iOS releases instead of staying grounded in the retrieved turns.
- **Possible fix**: Set generation temperature to 0.0 and add an automated check that flags any software version numbers not present in the retrieved evidence JSON.

### 5. Out-of-domain query with low similarity (Demo Example 5)
- **Customer**: *"Can you help me install Linux Ubuntu onto my Apple Watch Series 3?"*
- **Retrieved evidence**: Top match had similarity 0.1900 (below the 0.28 threshold).
- **What went wrong**: Without a threshold, the model would attempt to invent troubleshooting steps for an unsupported request.
- **Hypothesis**: The historical AppleSupport dataset contains zero conversations about installing Linux on an Apple Watch.
- **Fix implemented**: The escalation layer caught the low score (`insufficient_retrieval_grounding`) and routed the query to a human agent.

---

## 7. Next week

If given an additional week to improve this system:

1. **Expand human evaluation set**: Label 250 additional examples with dual-annotator review to measure Cohen's Kappa inter-annotator agreement.
2. **Pre-generation negation parser**: Extract phrases like "already tried X" or "restarted twice" and pass them as negative constraints so the generator stops repeating failed steps.
3. **Output validation layer**: Check that replies end with complete terminal punctuation and do not contain unauthorized commitment phrases before returning them.
4. **Calibrated classifier confidence**: Fit Platt scaling or isotonic regression on the Logistic Regression logits so confidence values represent true posterior probabilities.
5. **Cross-encoder reranker**: Add a lightweight reranker (`bge-reranker-base`) after FAISS retrieval to improve top-1 relevance on ambiguous queries.
6. **Scale FAISS index**: Expand from 2,000 to 10,000+ conversations using `IndexIVFFlat` to maintain sub-10ms search latency with broader topic coverage.
7. **Multi-turn dialogue evaluation**: Extend the pipeline to accept conversation history and evaluate multi-turn support interactions.

---

## 8. Reproduction

All core evaluations run locally in **under 2 minutes** without API keys.

### 1. Environment setup
```bash
git clone https://github.com/sumitnegii/AppleChatSupport.git
cd AppleChatSupport

# Python 3.10+ recommended
pip install -r requirements.txt
# Core dependencies: scikit-learn pandas sentence-transformers faiss-cpu pytest fastapi uvicorn requests python-dotenv
```

### 2. Run offline evaluations (No API keys required, ~90 seconds)
```bash
# Evaluate baseline intent classifiers (majority, TF-IDF holdout, CV)
python3 baseline_intent_eval.py

# Evaluate semantic embedding intent classifier
python3 semantic_intent_eval.py

# Run retrieval sanity check on FAISS index
python3 historical_retrieval.py

# Evaluate LLM judge agreement against human scores (using cached scores)
python3 llm_judge_eval.py --use-cache
```

### 3. Run automated tests (27 unit & smoke tests, ~20 seconds)
```bash
pytest -v
```

### 4. Run the interactive web demo
```bash
python3 server.py
# Open http://localhost:8000 in your browser
```
*Note: If no LLM API key is configured in `.env`, the demo automatically uses the deterministic grounded template fallback.*

### 5. Optional: Live LLM generation (Requires API key)
To generate replies with a live model, add your key to `.env`:
```bash
GROQ_API_KEY=your_key_here
# or GEMINI_API_KEY_MAIN=your_key_here
# or OPENAI_API_KEY=your_key_here
```
Then run:
```bash
python3 grounded_reply_generation.py
```

---

## 9. Limitations

- **Small retrieval index**: The 2,000-conversation FAISS index covers ~2.5% of the total dataset. Uncommon issues may lack matching historical cases.
- **Uncalibrated classifier probabilities**: The Logistic Regression output probabilities are not statistically calibrated. The 0.13 threshold is an engineering safeguard, not a formal probability cutoff.
- **Single-turn focus**: The agent currently processes the latest customer message. It does not maintain session state across long multi-turn exchanges.
- **Twitter format bias**: Historical support tweets are constrained to 140–280 characters and frequently ask users to switch to DM, which limits the depth of historical resolutions.
- **English only**: The pipeline filters for English messages and does not handle multilingual queries.

---

## 10. Project structure

```
.
├── README.md                              # Project documentation and evaluation report
├── DECISION_LOG.md                        # 14 engineering decisions and trade-offs
├── agent_pipeline.py                      # 6-stage unified support pipeline
├── baseline_intent_eval.py                # Majority and TF-IDF classifier benchmarks
├── semantic_intent_eval.py                # SentenceTransformer classifier benchmark
├── historical_retrieval.py                # FAISS vector index builder and search
├── grounded_reply_generation.py           # Retrieval-grounded reply generator
├── escalation_decision.py                 # Rule-based escalation decision engine
├── llm_judge_eval.py                      # LLM-as-a-judge and agreement metrics
├── server.py                              # FastAPI web demo server
├── static/
│   └── index.html                         # Interactive demo UI (vanilla HTML/JS)
├── golden_eval.csv                        # 250 hand-labelled golden evaluation examples
├── golden_candidates.csv                  # Candidate customer messages with weak labels
├── reply_quality_human_eval_scored.csv    # 30 human-scored reply evaluations (0-2 scale)
├── reply_quality_llm_judge_eval.csv       # LLM judge scores and justifications (N=30)
├── apple_support_faiss.index              # Local FAISS index (2,000 conversation vectors)
├── apple_support_faiss_mapping.json       # FAISS vector ID to conversation ID mapping
├── apple_support_conversation_index.csv   # Historical conversation index with embeddings
├── test_api_smoke.py                      # API endpoint smoke tests (5 tests)
├── test_escalation_smoke.py               # Escalation rule tests (9 tests)
├── test_llm_judge_smoke.py                # Judge rubric and metrics tests (7 tests)
├── test_retrieval_pipeline_smoke.py       # FAISS index and retrieval tests (5 tests)
└── test_grounded_reply_generation_smoke.py# Grounded generator smoke test (1 test)
```
