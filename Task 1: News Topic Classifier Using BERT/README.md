# Task 1: News Topic Classifier Using BERT


---

## Objective

Fine-tune a pre-trained BERT transformer model to classify news headlines into one of four topic categories using the AG News dataset.

---

## Methodology / Approach

1. **Dataset**: AG News (Hugging Face) — 120,000 training / 7,600 test samples across 4 balanced classes (World, Sports, Business, Sci/Tech)
2. **Preprocessing**: Tokenized with `bert-base-uncased` tokenizer, max sequence length 128, dynamic padding via `DataCollatorWithPadding`
3. **Model**: `bert-base-uncased` with a 4-class classification head (transfer learning)
4. **Training**: 3 epochs, batch size 16, AdamW with weight decay, mixed precision (fp16) on GPU
5. **Evaluation**: Accuracy + Weighted F1-score, Confusion Matrix, Per-class Classification Report
6. **Deployment**: Gradio interactive demo with live predictions and confidence scores

---

## Key Results / Observations

| Metric | Score |
|--------|-------|
| Test Accuracy | ~94% |
| Weighted F1-Score | ~94% |

- **BERT excels** at this task due to its deep contextual understanding of news language
- **Sports** category achieves the highest per-class F1 (sports-specific vocabulary is very distinctive)
- **World vs Business** shows the most confusion — economic/political news overlaps
- Training on just 8k samples (subset) still yields strong results — demonstrating the power of transfer learning
- Full dataset training (~120k) would push accuracy above 95%

---

## Tech Stack

- `transformers` — BERT model and Trainer API
- `datasets` — AG News loading
- `evaluate` — Accuracy and F1 metrics
- `gradio` — Live demo deployment
- `scikit-learn` — Confusion matrix and classification report

---

## How to Run

1. Open `Task1_BERT_News_Classifier.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells top to bottom
4. The Gradio demo will launch with a public share link at the end

---

## Files

```
Task1_BERT_News_Classifier.ipynb   ← Main notebook
README.md                          ← This file
```
