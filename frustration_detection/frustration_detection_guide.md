
# 🧠 Frustration Detection in Dialogues with Transformers

## 🧩 Problem Background

Frustration is a key emotional signal in human-computer interactions. In task-oriented dialogue systems, especially those powered by LLMs, users often feel frustrated when:
- Their intent is misunderstood,
- The assistant loops or fails to act,
- Answers are irrelevant, generic, or hallucinated.

Detecting frustration allows systems to:
- Escalate to human agents,
- Rephrase or clarify responses,
- Improve long-term user satisfaction.

While emotion detection in text has matured (e.g. detecting anger, sadness, etc.), **frustration** is more subtle and requires **context-aware models**. This guide walks you through building a **binary classifier (Frustrated vs Not Frustrated)** using **transformer-based models** on the **EmpatheticDialogues** dataset.

---

## 🛠 Implementation Guide

### 1. Install Dependencies

```bash
pip install transformers datasets scikit-learn
```

---

### 2. Load the Dataset

Use Hugging Face's `facebook/empathetic_dialogues`. It contains 25k conversations, each labeled with one of 32 emotion labels.

```python
from datasets import load_dataset
dataset = load_dataset("facebook/empathetic_dialogues")
```

---

### 3. Define Frustration Mapping

Define which of the 32 emotions are considered "frustrated".

```python
FRUSTRATED_LABELS = {
    "annoyed", "angry", "disgusted", "jealous", "embarrassed", 
    "ashamed", "afraid", "terrified", "furious", "devastated", "disappointed"
}
```

---

### 4. Map to Binary Labels

```python
def map_to_binary(example):
    return {
        "text": example["prompt"] + " " + " ".join(example["utterance"]),
        "label": int(example["context"] in FRUSTRATED_LABELS)
    }

dataset = dataset.map(map_to_binary)
```

---

### 5. Tokenize

```python
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

---

### 6. Define Model and Training Setup

Use a RoBERTa model fine-tuned on emotions:

```python
from transformers import RobertaForSequenceClassification

model = RobertaForSequenceClassification.from_pretrained(
    "SamLowe/roberta-base-go_emotions",
    num_labels=2
)
```

---

### 7. Add Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight
import torch

train_labels = [int(x["label"]) for x in tokenized["train"]]
class_weights = compute_class_weight("balanced", classes=[0,1], y=train_labels)
loss_weights = torch.tensor(class_weights, dtype=torch.float)
```

---

### 8. Train with Trainer API

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

Add evaluation metrics:

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

Then run training:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
```

---

### 9. Evaluate

```python
results = trainer.evaluate(tokenized["test"])
print(results)
```

---

## ✅ Tips for Better Performance

- Provide **full dialog context** (prompt + utterances).
- Use **class weighting** to avoid bias.
- Fine-tune **hyperparameters** (use Optuna).
- Optionally reduce to 6 or 2 emotion categories.

---

## 📚 Next Steps

- Train on your own chat logs.
- Apply to frustration forecasting (not just detection).
- Use LLMs like GPT-4o in in-context learning mode to label frustration in unlabeled data.

---

## 📩 Contact

If you need help adapting this to your own dataset, or deploying the model in production, feel free to reach out.
