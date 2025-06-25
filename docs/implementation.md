# Implementation Brief: **EmoWOZ-Ahead (Text-Only) Benchmark**

> **Goal:** build and evaluate models that predict—**one user turn in advance**—whether the next user utterance in a task-oriented dialogue will be **“dissatisfied / frustrated.”**

---

## 1. Dataset Preparation

| Step                           | Command / Key Points                                                                                                                                |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1.1 Install & load EmoWOZ**  | `pip install datasets`  <br>`from datasets import load_dataset`  <br>`emo = load_dataset("hhu-dsml/emowoz")`                                        |
| **1.2 Shift labels (+1 turn)** | Use the provided script `make_shifted_labels.py`. <br>For every user turn *t*, add field<br>`willBeFrustrated = int(emotion(t+1)=="dissatisfied")`. |
| **1.3 Filter & format**        | Keep only **user** turns.  <br>For each example store: <br>`{"context": [u_{t-N+1}, …, s_t], "label": 0/1}`.                                        |
| **1.4 Splits**                 | Respect original **train / valid / test** user-ID splits.                                                                                           |
| **1.5 Stats check**            | Expect ≈ 6–8 % positives per split; log counts.                                                                                                     |
| **1.6 Save**                   | JSONL files: `train.jsonl`, `val.jsonl`, `test.jsonl`.                                                                                              |

---

## 2. Repository Skeleton

```
emowoz-ahead/
 ├─ data_scripts/
 │   └─ make_shifted_labels.py
 ├─ data/            # output JSONL splits
 ├─ baselines/
 │   ├─ bert_cls.py  # last-utterance model
 │   └─ context_gru.py
 ├─ models/
 │   └─ dialo_gpt_small.py
 ├─ eval.py          # macro-F1, AUC, latency
 ├─ README.md        # benchmark card & usage
 ├─ LICENSE          # CC-BY-4.0 data, Apache-2 code
 └─ requirements.txt
```

---

## 3. Models to Implement

| Tag    | Context Window                     | Architecture                                      | Expected Macro-F1 |
| ------ | ---------------------------------- | ------------------------------------------------- | ----------------- |
| **M0** | –                                  | Majority-0                                        | 0.00              |
| **M1** | 1 user turn                        | `bert-base-uncased` CLS                           | \~0.23            |
| **M2** | Last **3** user+sys turns (concat) | `roberta-base` CLS                                | \~0.27            |
| **M3** | Same 3-turn context                | RoBERTa embeddings → **1-layer GRU** (128-d) → FC | \~0.30            |
| **M4** | **5** turns                        | `DialoGPT-small` encoder fine-tuned               | ≥0.35 target      |

**Training defaults**

```
epochs: 3
optimizer: AdamW
lr: 2e-5
batch: 16
loss: BCEWithLogits
class_weight: {0:1, 1:12}  # adjust after stats check
early_stop: patience=2 on macro-F1(dev)
```

---

## 4. Evaluation Script (`eval.py`)

```bash
python eval.py \
  --model_path checkpoints/M3 \
  --test_file data/test.jsonl \
  --metrics macro_f1 auc latency
```

Outputs JSON:

```json
{
  "macro_f1": 0.312,
  "auc": 0.74,
  "latency_ms": 8.1
}
```

*Measure latency* over GPU **and** CPU (`time.perf_counter()` around forward pass).

---

## 5. Deliverables & Timeline

| Day | Deliverable                                               |
| --- | --------------------------------------------------------- |
| 1   | Dataset JSONL + class‐balance report                      |
| 2   | M1 baseline training & test metrics                       |
| 3   | M2 implementation + metrics                               |
| 4   | M3 (GRU) training + ablation (context window N = 1, 3, 5) |
| 5   | M4 (DialoGPT) fine-tune + final comparison table          |
| 6   | `eval.py` latency benchmark + README usage guide          |
| 7   | Error analysis notebook (top FP/FN examples)              |

---

## 6. Success Criteria

* **Macro-F1 ≥ 0.30** on test with M3 or better.
* **Latency ≤ 15 ms** per prediction (GPU).
* Clean, reproducible code; README lets others replicate in ≤ 10 min.
