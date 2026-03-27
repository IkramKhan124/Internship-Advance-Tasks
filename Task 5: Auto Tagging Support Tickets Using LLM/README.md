# Task 5: Auto Tagging Support Tickets Using LLM


---

## Objective

Automatically classify and tag customer support tickets into predefined categories using prompt engineering with a Large Language Model, comparing zero-shot vs few-shot learning approaches.

---

## Methodology / Approach

1. **Dataset**: Synthetic support ticket dataset — 55+ tickets across 10 categories (Billing, Technical, Account, Shipping, Returns, Bug Report, Feature Request, Password Reset, Cancellation, General Inquiry)

2. **Model**: `google/flan-t5-large` — free, no API key required, strong instruction-following capability

3. **Approach 1 — Zero-Shot**:
   - Prompt includes only the task description and category list
   - No examples provided; model relies purely on pre-training knowledge
   - Asks for top 3 most probable categories

4. **Approach 2 — Few-Shot**:
   - Prompt includes 2 labeled examples per category (20 total examples)
   - Model uses in-context learning to understand the pattern and expected output format
   - Same top-3 output structure

5. **Evaluation**: Top-1 accuracy, Top-3 accuracy, per-category F1-score, confusion matrix

6. **Output**: Top 3 ranked tags per ticket with correct parsing and fallback handling

---

## Observations


- **Few-shot improves accuracy by ~15–20%** over zero-shot — examples are crucial
- **Top-3 accuracy is near 90%+** — the correct tag is almost always in the top predictions
- **Password Reset and Cancellation** are the easiest to classify (distinctive vocabulary)
- **General Inquiry vs others** is the primary confusion case — vague tickets look like inquiries
- **Ambiguous tickets** get multiple relevant tags in top-3, which is actually useful for routing

---

## Tech Stack

- `transformers` — FLAN-T5 Large model
- `torch` — inference
- `scikit-learn` — accuracy, F1, confusion matrix
- `pandas` / `numpy` — data handling
- `matplotlib` / `seaborn` — visualizations

---

## How to Run

1. Open `Task5_Auto_Tagging.ipynb` in Google Colab
2. GPU runtime recommended (T4) — speeds up FLAN-T5 inference significantly
3. Run all cells — dataset is generated automatically

---

## Extending to Production

- Replace FLAN-T5 with **GPT-4o-mini** or **Claude Haiku** for near-perfect accuracy at low cost
- Add **fine-tuning** on labeled company tickets for domain-specific improvements
- Integrate with ticketing systems (Zendesk, Freshdesk) via API for automated routing

---

## Files

```
Task5_Auto_Tagging.ipynb   ← Main notebook
README.md                   ← This file
```
