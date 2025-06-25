# EmoWOZ-Ahead Implementation Plan
**One-Turn-Ahead Frustration Forecasting in Task-Oriented Dialogs**

---

## 📋 Project Overview
**Goal**: Build the first public benchmark for predicting user frustration one turn ahead in task-oriented dialogues using the EmoWOZ dataset.

**Success Criteria**:
- Macro-F1 ≥ 0.30 on test set
- Inference latency ≤ 15ms (GPU)
- Reproducible benchmark with clean code
- Ready for HRI 2026 submission

---

## 🗓️ 7-Day Implementation Timeline

### **Day 1: Project Setup & Data Preparation**
#### Step 1.1: Environment Setup
```bash
# Create project directory
mkdir emowoz-ahead
cd emowoz-ahead

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install datasets transformers torch scikit-learn pandas numpy matplotlib seaborn jupyter tqdm
pip freeze > requirements.txt
```

#### Step 1.2: Repository Structure
```bash
mkdir -p data_scripts data baselines models checkpoints results notebooks
touch README.md LICENSE eval.py
```

#### Step 1.3: Load EmoWOZ Dataset
Create `data_scripts/load_emowoz.py`:
```python
from datasets import load_dataset
import pandas as pd
import json

# Load the dataset
emo = load_dataset("hhu-dsml/emowoz")
print("Dataset loaded successfully")
print(f"Train: {len(emo['train'])}")
print(f"Validation: {len(emo['validation'])} 
print(f"Test: {len(emo['test'])}")

# Explore structure
print("\nSample entry:", emo['train'][0])
```

#### Step 1.4: Implement Label Shifting
Create `data_scripts/make_shifted_labels.py`:
```python
def shift_labels_forward(dataset_split):
    """
    Shift emotion labels forward by one user turn.
    For user turn t, predict if turn t+1 will be 'dissatisfied'
    """
    shifted_data = []
    
    for dialogue in dataset_split:
        turns = dialogue['turns']
        user_turns = [t for t in turns if t['speaker'] == 'user']
        
        for i in range(len(user_turns) - 1):
            current_turn = user_turns[i]
            next_turn = user_turns[i + 1]
            
            # Create context (last N turns including system responses)
            context = extract_context(turns, current_turn['turn_id'], N=3)
            
            # Label: 1 if next user turn is dissatisfied, 0 otherwise
            label = 1 if next_turn['emotion'] == 'dissatisfied' else 0
            
            shifted_data.append({
                'dialogue_id': dialogue['dialogue_id'],
                'turn_id': current_turn['turn_id'],
                'context': context,
                'label': label,
                'text': current_turn['text']
            })
    
    return shifted_data

def extract_context(turns, current_turn_id, N=3):
    """Extract last N turns as context"""
    # Implementation here
    pass
```

#### Step 1.5: Generate Dataset Splits
Create `data_scripts/create_splits.py` and run:
```bash
python data_scripts/create_splits.py
```

Expected output files:
- `data/train.jsonl`
- `data/val.jsonl` 
- `data/test.jsonl`
- `data/dataset_stats.json`

#### Step 1.6: Class Balance Analysis
Check positive class ratio (expect 6-8%):
```python
# Add to create_splits.py
def analyze_class_balance(data):
    total = len(data)
    positive = sum(item['label'] for item in data)
    print(f"Total samples: {total}")
    print(f"Positive samples: {positive} ({positive/total*100:.2f}%)")
    return positive/total
```

---

### **Day 2: Baseline M1 - BERT Single Turn**
#### Step 2.1: Implement BERT-CLS Model
Create `baselines/bert_cls.py`:
```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertCLS(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
```

#### Step 2.2: Training Script
Create `baselines/train_bert.py`:
```python
# Training configuration
config = {
    'model_name': 'bert-base-uncased',
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'epochs': 3,
    'class_weights': [1.0, 12.0],  # Adjust based on class imbalance
    'patience': 2
}
```

#### Step 2.3: Run Training
```bash
python baselines/train_bert.py --config_file config/bert_config.json
```

#### Step 2.4: Evaluate M1
Expected metrics:
- Macro-F1: ~0.23
- Save results to `results/M1_bert_results.json`

---

### **Day 3: Baseline M2 - RoBERTa Context**
#### Step 3.1: Implement RoBERTa-CLS
Create `baselines/roberta_cls.py`:
```python
from transformers import RobertaModel, RobertaTokenizer

class RobertaCLS(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)
```

#### Step 3.2: Context Window Processing
Modify data loading to concatenate last 3 user+system turns:
```python
def create_context_string(context_turns):
    """Concatenate turns with special tokens"""
    context_str = ""
    for turn in context_turns[-3:]:  # Last 3 turns
        speaker = "USER" if turn['speaker'] == 'user' else "SYSTEM"
        context_str += f"[{speaker}] {turn['text']} "
    return context_str.strip()
```

#### Step 3.3: Train and Evaluate M2
Expected metrics:
- Macro-F1: ~0.27
- Compare with M1 results

---

### **Day 4: Baseline M3 - RoBERTa + GRU**
#### Step 4.1: Implement Temporal Model
Create `baselines/roberta_gru.py`:
```python
class RobertaGRU(nn.Module):
    def __init__(self, roberta_model='roberta-base', gru_hidden=128):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.gru = nn.GRU(
            input_size=self.roberta.config.hidden_size,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Linear(gru_hidden, 1)
        
    def forward(self, input_ids, attention_mask):
        # Process each turn separately through RoBERTa
        batch_size, seq_len, max_len = input_ids.shape
        
        # Reshape for processing
        input_ids = input_ids.view(-1, max_len)
        attention_mask = attention_mask.view(-1, max_len)
        
        # Get embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output
        
        # Reshape back to sequence
        embeddings = embeddings.view(batch_size, seq_len, -1)
        
        # Process through GRU
        gru_output, _ = self.gru(embeddings)
        final_hidden = gru_output[:, -1, :]  # Last hidden state
        
        return self.classifier(final_hidden)
```

#### Step 4.2: Ablation Study - Context Window Size
Test with N = {1, 3, 5} turns:
```bash
python baselines/train_roberta_gru.py --context_window 1
python baselines/train_roberta_gru.py --context_window 3  
python baselines/train_roberta_gru.py --context_window 5
```

#### Step 4.3: Target Achievement
Goal: Macro-F1 ≥ 0.30
Expected: ~0.30-0.32

---

### **Day 5: Baseline M4 - DialoGPT**
#### Step 5.1: Implement DialoGPT Model
Create `models/dialo_gpt_small.py`:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class DialoGPTClassifier(nn.Module):
    def __init__(self, model_name='microsoft/DialoGPT-small'):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.gpt.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use last token representation
        last_hidden = outputs.last_hidden_state[:, -1, :]
        return self.classifier(last_hidden)
```

#### Step 5.2: Fine-tuning Setup
```python
config = {
    'model_name': 'microsoft/DialoGPT-small',
    'max_length': 1024,  # Longer context
    'batch_size': 8,     # Smaller due to memory
    'learning_rate': 1e-5,
    'epochs': 5,
    'context_window': 5
}
```

#### Step 5.3: Target Performance
Goal: Macro-F1 ≥ 0.35 (best model)

---

### **Day 6: Evaluation & Benchmarking**
#### Step 6.1: Implement Comprehensive Evaluation
Create `eval.py`:
```python
import time
import torch
from sklearn.metrics import f1_score, roc_auc_score, classification_report

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    labels = []
    latencies = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Measure latency
            start_time = time.perf_counter()
            
            outputs = model(**batch)
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # ms
            
            # Collect predictions
            preds = torch.sigmoid(outputs.logits) > 0.5
            predictions.extend(preds.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    macro_f1 = f1_score(labels, predictions, average='macro')
    auc = roc_auc_score(labels, predictions)
    avg_latency = sum(latencies) / len(latencies)
    
    return {
        'macro_f1': macro_f1,
        'auc': auc,
        'latency_ms': avg_latency,
        'classification_report': classification_report(labels, predictions)
    }
```

#### Step 6.2: Benchmark All Models
```bash
python eval.py --model_path checkpoints/M1 --test_file data/test.jsonl
python eval.py --model_path checkpoints/M2 --test_file data/test.jsonl  
python eval.py --model_path checkpoints/M3 --test_file data/test.jsonl
python eval.py --model_path checkpoints/M4 --test_file data/test.jsonl
```

#### Step 6.3: Generate Comparison Table
Create `results/benchmark_comparison.json`:
```json
{
  "M1_BERT": {"macro_f1": 0.23, "auc": 0.71, "latency_ms": 5.2},
  "M2_RoBERTa": {"macro_f1": 0.27, "auc": 0.73, "latency_ms": 6.8},
  "M3_RoBERTa_GRU": {"macro_f1": 0.31, "auc": 0.76, "latency_ms": 9.1},
  "M4_DialoGPT": {"macro_f1": 0.36, "auc": 0.78, "latency_ms": 12.4}
}
```

---

### **Day 7: Documentation & Error Analysis**
#### Step 7.1: Create Benchmark README
Update `README.md` with:
- Dataset description
- Model architectures
- Usage instructions
- Reproduction guide
- Citation information

#### Step 7.2: Error Analysis Notebook
Create `notebooks/error_analysis.ipynb`:
```python
# Analyze false positives and false negatives
# Extract top confusing examples
# Visualize prediction confidence distributions
# Context length vs performance analysis
```

#### Step 7.3: Dataset Card
Create `data/dataset_card.md`:
- Data source and construction
- Label distribution
- Ethical considerations
- Intended use

#### Step 7.4: Final Package
```bash
# Ensure reproducibility
pip freeze > requirements.txt

# License files
cp LICENSE data/
cp LICENSE models/

# Create release archive
tar -czf emowoz-ahead-benchmark.tar.gz data/ models/ baselines/ README.md LICENSE requirements.txt
```

---

## 🎯 Key Deliverables Checklist

### Data & Preprocessing
- [ ] EmoWOZ dataset loaded and explored
- [ ] Label shifting implemented and validated
- [ ] Train/val/test splits created (respect original user-ID splits)
- [ ] Class balance analysis completed (~6-8% positive)
- [ ] JSONL files generated and validated

### Models & Training
- [ ] M1: BERT-CLS implemented and trained
- [ ] M2: RoBERTa-CLS with context implemented  
- [ ] M3: RoBERTa+GRU with temporal modeling
- [ ] M4: DialoGPT fine-tuned model
- [ ] All models achieve target latency < 15ms

### Evaluation & Analysis
- [ ] Comprehensive evaluation script (`eval.py`)
- [ ] Macro-F1 ≥ 0.30 achieved with at least one model
- [ ] Latency benchmarking on GPU and CPU
- [ ] Error analysis and failure case study
- [ ] Context window ablation study

### Documentation & Reproducibility
- [ ] Clean, documented codebase
- [ ] README with usage instructions
- [ ] Dataset card and ethical considerations
- [ ] Requirements.txt and dependency management
- [ ] CC-BY-4.0 license for data, Apache-2 for code

### Research Contribution
- [ ] First public benchmark for one-turn-ahead frustration
- [ ] Baseline results for future comparison
- [ ] Code and data ready for community use
- [ ] Draft ready for HRI 2026 submission

---

## 🚀 Quick Start Commands

```bash
# Day 1: Setup
git clone <repo> && cd emowoz-ahead
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python data_scripts/create_splits.py

# Day 2-5: Train models
python baselines/train_bert.py
python baselines/train_roberta_cls.py  
python baselines/train_roberta_gru.py
python models/train_dialo_gpt.py

# Day 6: Evaluate
python eval.py --all_models

# Day 7: Package
jupyter notebook notebooks/error_analysis.ipynb
tar -czf emowoz-ahead-benchmark.tar.gz data/ models/ baselines/ README.md
```

---

## 📊 Expected Results Summary

| Model | Context | Macro-F1 | AUC | Latency (ms) |
|-------|---------|----------|-----|--------------|
| M1: BERT-CLS | 1 turn | ~0.23 | ~0.71 | ~5 |
| M2: RoBERTa-CLS | 3 turns | ~0.27 | ~0.73 | ~7 |
| M3: RoBERTa+GRU | 3 turns | ~0.31 | ~0.76 | ~9 |
| M4: DialoGPT | 5 turns | ~0.36 | ~0.78 | ~12 |

**Success threshold**: Any model with Macro-F1 ≥ 0.30 and latency ≤ 15ms constitutes a successful benchmark contribution. 