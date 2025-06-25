# Frustration Prediction in Task-Oriented Dialogs

## User Frustration Prediction in Task-Oriented Dialogue Systems  
**Research Proposal**  
_Omar Hammad_

---

## Background

- **Task-Oriented Dialogue Systems (TODS)** are conversational agents specifically designed to help users complete defined tasks—for instance, booking a flight, ordering food, or troubleshooting a product (Lee et al., 2021).
- **Frustration** is an emotional response resulting from an obstacle preventing the satisfaction of a need (Berkowitz, 1989).
- **Sources of frustration**: low system performance, limited usefulness, poor usability (Hertzum and Hornbæk, 2023), goal blockage, time loss, frequent breakdowns, unexpected system behavior, and user-specific traits (Lazar, 2006).
- **Importance**: Predicting frustration is key for maintaining user satisfaction, engagement, and retention (Hernandez Caralt et al., 2025).

---

## Human Frustration – Broader Factors

- **Conversation norms**: expectation violations, repeated goal blockage, time loss.
- **Interaction dynamics**: frequent interruptions, conversational dominance.
- **Cognitive/contextual load**: task complexity, low relational trust.
- **Momentary affect**: irritability, boredom amplifies frustration.
- **Demographics/Culture**: age, gender expressivity, cultural norms.

---

## Project Idea

- Build an ML pipeline to predict user frustration in TODS.
- Establish a new benchmark for frustration prediction in TODS.
- Customize models for different contexts (e.g. small models, task-specific).

---

## Recent Related Work

| Year | Authors | Emotion | Method | Dataset | Score | Limits |
|------|---------|---------|--------|---------|-------|--------|
| 2020 | Zuters & Leonova | Frustration intensity | BoW, emoji → MLP ensemble | Twitter CS threads | Top-1 Acc 46% | No split released |
| 2023 | Li et al. | 6 emotions incl. frustration | 2-layer Transformer | WoZ robot dialogs (JP) | Macro-F1 0.46 | No English TOD, private |
| 2023 | Altarawneh et al. | 7 emotions | GCN | MELD | Macro-F1 0.43 | Scripted TV |
| 2024 | Koga et al. | Implicit partner emotion | BERT + COMET | DailyDialog, MELD | Macro-F1 0.39 | Weak on frustration |
| 2025 | Telepathy Labs | Frustration (same-turn) | GPT-4 prompt | Booking-bot logs | F1 0.58 | Proprietary |

---

## Related Work – Observations

- No reproducible benchmark for one-turn-ahead task-oriented frustration.
- Performance ceiling remains low (Macro-F1 ≤ 0.46).
- Frustration often diluted in broader “negative” classes.
- Telepathy (2025) is the only one isolating frustration.

---

## Related Work – Methods

- Context-window LMs: BERT/RoBERTa on 1-4 turns. No GPT finetuning.
- Prompt-only GPT: No training, prompt feeding only (Telepathy Labs).
- Context windows short (≤ 5 turns). Runtime rarely reported.

---

## Related Work – Datasets

- **EmoWOZ**: Task-oriented, dissatisfaction label.
- **HSRI (2025)**: Real human-robot data, annotations.
- **MELD**: Scripted, no frustration label.
- **DailyDialog**: Scripted chit-chat, no dissatisfaction tag.
- **Telepathy booking-bot**: Real, proprietary.
- Others: WoZ Robot (JP), Twitter CS, IVR corpora.

---

## Research Gap and Questions

- **Gap**: No open TOD benchmark surpassing Macro-F1 0.46.
- **RQ1**: Can we predict future user frustration using only TOD history?
- **RQ2**: Does adding facial expressions improve over text-only?
- **RQ3**: Do user-persona vectors enhance forecasting?
- **RQ4**: What’s the smallest real-time model maintaining ≥90% F1?
- **RQ5**: Can TOD-trained models transfer to open-domain settings?

---

## Proposed Method – Data & Context

- **Dataset**: EmoWOZ
- **Label shift**: Advance "dissatisfied" by one turn ⇒ “will-be-frustrated”
- **Splits**: 80/10/10, dialog-level
- **Context windows**: Last N = {1, 3, 5} turns

---

## Proposed Method – Model Architecture

- **Turn encoder**: RoBERTa-base → CLS
- **Temporal aggregator**: 2-layer GRU (256h)
- **User-feature block**:
  - Expectation mismatch (cosine)
  - Interaction metrics (interruptions, dominance, gap)
  - Cognitive proxy: F-K grade, perplexity
  - Trust/mood: hedging, valence

- **Fusion**: [GRU | features] → LayerNorm → MLP (256→64→1, GELU, dropout) → sigmoid

---

## Proposed Method – Training & Evaluation

- **Loss**: class-weighted BCE
- **Optimizer**: AdamW, LR=2e-5, 3–5 epochs
- **Metrics**: Macro-F1
- **Baselines**:
  - Majority-0
  - BERT-CLS (N=1)
  - RoBERTa-CLS (N=3)
  - RoBERTa + GRU (no user features)
- **Ablations**:
  - Drop user-features
  - Vary N
  - Remove system turns

---

## Future Directions

- Include more modalities
- Expand emotion set
- Advanced user modeling
- Optimize for speed
- Apply to robot settings
- Transfer beyond TODs

---

## Next Steps

- Literature review
- Design the study
- Conduct experiments
- Analyze data
- Plan future work
