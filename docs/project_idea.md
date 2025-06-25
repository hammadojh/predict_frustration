## Scientific Brief – *EmoWOZ-Ahead: One-Turn-Ahead Frustration Forecasting*

### 1   Background

Conversational AI has made solid progress in *recognising* a user’s **current** emotion, yet nearly all commercial and research systems respond **after** frustration surfaces. Recent academic work (e.g., Li et al., 2023; Koga et al., 2024) shows that short-horizon *emotion forecasting* is feasible, but reported F1 scores remain below 0.50, and **no public benchmark** targets binary “frustrated next turn” in task-oriented text dialogue.

### 2   Research Gap

* Existing datasets (EmoWOZ, DailyDialog, MELD) label *current* emotions.
* Anticipatory studies are limited to small acted corpora or prosody-only signals.
* Industry white-papers call for **early warning** to trigger repair or human hand-off, but provide no reproducible split or baseline.

### 3   Goal (This Project)

Create the first **public, text-only benchmark**—*EmoWOZ-Ahead*—and establish strong neural baselines that **predict if the next user utterance will be tagged “dissatisfied.”**
Practical payoff: a proactive alarm the robot or chatbot can act upon before escalation.

### 4   Formal Problem Statement

Given a dialogue history of length $N$ turns

$$
\mathcal{C}_t = \big[u_{t-N+1}, s_{t-N+1},\ldots,u_t,s_t\big],
$$

predict the binary label

$$
y_t = \mathbf{1}\!\{\,\text{emotion}(u_{t+1}) = \text{“dissatisfied”}\,\}.
$$

We learn a parameterised function $f_\theta(\mathcal{C}_t)=\hat{y}_t$ and optimise the binary cross-entropy

$$
\mathcal{L}(\theta)= -\,\big[y_t\log\hat{y}_t+(1-y_t)\log(1-\hat{y}_t)\big].
$$

**Primary metric:** Macro-F1 over classes {frustration-next, no-frustration}.
**Secondary:** ROC-AUC and inference latency.

### 5   Dataset Construction

* Start from **EmoWOZ** (task-oriented English booking dialogues).
* **Shift labels forward by one user turn** to obtain $y_t$ (≈ 6–8 % positives).
* Keep original speaker-segregated train/valid/test splits → reproducible benchmark **“EmoWOZ-Ahead”.**

### 6   Baseline Suite

| Model          | Context window   | Rationale                      |
| -------------- | ---------------- | ------------------------------ |
| Majority-0     | –                | Trivial lower bound            |
| BERT-CLS       | 1 user turn      | Tests single-utterance cues    |
| RoBERTa-CLS    | 3 user+sys turns | Adds dialogue context          |
| RoBERTa + GRU  | 3–5 turns        | Captures temporal order        |
| DialoGPT-small | 5 turns          | Leverages dialogue-pretraining |

A Macro-F1 ≥ 0.30 already surpasses earlier works in similar settings and would constitute a publishable baseline.

### 7   Expected Scientific Contribution

1. **Benchmark** – first open split for one-turn-ahead frustration.
2. **Empirical insight** – quantifies how much textual context alone can anticipate dissatisfaction.
3. **Foundation** – paves the way for multimodal extensions (prosody, facial cues) in future work.

### 8   Success Criteria

* Dataset card + code released under CC-BY-4.0.
* At least one model ≥ 0.30 Macro-F1, < 15 ms latency (GPU).
* Short paper draft ready for submission to HRI 2026 Late-Breaking Reports.

---

> **Your mission as data scientist:** implement the data shift, train the listed baselines, report metrics, and package a clean, reproducible benchmark so the research team can focus on analysis and publication.
