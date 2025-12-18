# Banking77 Intent Classification with DistilBERT & Gemini Augmentation

This project focuses on fine-tuning a **DistilBERT** model to classify customer service queries into **77 distinct banking intents** using the **Banking77 dataset**.  
To address class imbalance and improve model robustness, a **data augmentation pipeline** was developed using **Google Gemini 2.5 Flash**.

The final model achieves a **Test Accuracy of 92.63%**.

---

## Project Overview

Customer service automation requires understanding very specific user intents.  
The Banking77 dataset presents a fine-grained intent classification challenge with 77 categories such as:

- `lost_or_stolen_card`
- `exchange_rate`
- `declined_transfer`

### Key Features

- **LLM-Based Data Augmentation**  
  Gemini API was used to generate synthetic samples for underrepresented classes (fewer than 100 samples).  
  Generated samples vary in tone and style (formal, casual, slang) while preserving intent.

- **Efficient Fine-Tuning**  
  The model was trained on consumer-grade hardware (RTX 3050 6GB) using:
  - Mixed precision (FP16)
  - Gradient accumulation

- **High Performance**  
  Achieved over **92% accuracy** on the unseen test set.

---

## Project Structure

```bash
├── dataset/
│   ├── train.csv           # Original + augmented training data
│   └── test.csv            # Test data
├── banking77_final_model/
│   └── checkpoint-4350/    # Best performing model
├── preprocessing.ipynb     # Data analysis, cleaning, Gemini augmentation
├── training.ipynb          # Model training (Trainer API)
└── evaluation.ipynb        # Inference and test evaluation
```

## Data Preprocessing & Augmentation

The original dataset contained significant class imbalance.

Traditional text augmentation methods (e.g., synonym replacement with `nlpaug`) produced low-quality and unnatural sentences.  
To overcome this limitation, a **semantic augmentation pipeline** using **Google Gemini** was implemented.

### Augmentation Pipeline

**Analysis**  
Intent classes with fewer than 100 training samples were identified.

**Prompt Engineering**  
A strict system prompt was designed to ensure that:
- The original intent is preserved exactly
- Sentence structure and phrasing vary
- Banking context remains realistic and natural

**Result**  
The dataset was balanced, allowing the model to learn robust representations for all 77 intent classes.


---

## Training Configuration

The model was trained using the **Hugging Face Trainer API** with memory-efficient settings.

- **Base Model:** `distilbert-base-uncased`
- **Train Batch Size:** 16
- **Evaluation Batch Size:** 32
- **Gradient Accumulation:** 2  
  (Effective batch size = 32)
- **Learning Rate:** 2e-5
- **Epochs:** 15 (with early stopping)
- **Precision:** FP16 (mixed precision)

---

## Results

Training converged smoothly and stabilized around epoch 13.

| Metric               | Score   |
|----------------------|---------|
| Validation Accuracy  | ~91.60% |
| Validation Loss      | 0.37    |
| Test Accuracy        | 92.63%  |
| Test Loss            | 0.3124  |

---

## Inference Examples

The trained model produces high-confidence predictions for real-world banking queries.

| User Query                          | Predicted Intent        | Confidence |
|-----------------------------------|-------------------------|------------|
| "I lost my card, please help!"    | lost_or_stolen_card     | 99.37%     |
| "Why is my transfer declined?"    | declined_transfer       | 99.52%     |
| "Can I get a virtual card?"       | getting_virtual_card    | 81.01%     |

---

## Installation & Usage

### Requirements

```bash
pip install transformers datasets evaluate torch pandas google-generativeai python-dotenv
```
