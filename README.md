# Multi-Stage Suicide Risk Detection via Cognitive & Topic-Aware Models

A multi-stage NLP-based risk detection system developed as part of an MSc Data Science dissertation at VIT Chennai.

## Overview

This project explores a hierarchical approach for detecting and categorizing risk-related language from text. The system combines text preprocessing, transformer-based sentence embeddings, classical machine learning, and transformer-based classification to support real-time text risk analysis.

The proposed pipeline consists of two main stages:

1. **Binary Classification** – separates text into **No Risk** and **Risk** categories.
2. **Multiclass Classification** – further categorizes risky content into:
   - High Risk
   - Medium Risk
   - Low Risk
   - No Risk

The project also includes paragraph-level prediction, model evaluation, robustness testing, and Integrated Gradients-based explainability.

## Project Pipeline

```text
Raw Text / Datasets
        │
        ▼
Text Cleaning & Normalization
        │
        ├── Unicode normalization
        ├── HTML / URL removal
        ├── Special-character cleaning
        ├── Emoji normalization
        └── Whitespace normalization
        │
        ▼
MiniLM Sentence Embeddings
        │
        ▼
┌─────────────────────────────────┐
│        Stage 1: Binary          │
│   No Risk vs Risk Detection     │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│      Stage 2: Multiclass        │
│ High / Medium / Low / No Risk   │
└─────────────────────────────────┘
        │
        ▼
Paragraph-Level Aggregation
        │
        ▼
Prediction + Confidence
        │
        ▼
Logging / Analysis
```

## Key Features

- Text cleaning and normalization for noisy online language.
- Emoji and non-standard character handling.
- MiniLM-based 384-dimensional sentence embeddings.
- TF-IDF feature extraction for classical ML baselines.
- Risk lexicon features for supplementary linguistic information.
- LightGBM binary and multiclass classification.
- Logistic Regression TF-IDF baseline.
- Fine-tuned RoBERTa binary classification.
- Fine-tuned RoBERTa multiclass classification.
- Paragraph-level prediction using chunking and prediction aggregation.
- Confusion matrices and classification reports for evaluation.
- Integrated Gradients for model explainability.
- Robustness testing using controlled text noise.
- CSV-based prediction logging.

## Models

### 1. TF-IDF + Logistic Regression

Used as a classical baseline for multiclass risk classification.

### 2. TF-IDF + LightGBM

LightGBM was used for both binary and multiclass classification.

The report describes LightGBM as a lightweight model suitable for fast inference and high-dimensional data.

### 3. MiniLM + LightGBM

MiniLM transformer embeddings were generated from cleaned text and used as input features for LightGBM models.

### 4. RoBERTa Binary Classifier

A fine-tuned RoBERTa model performs binary classification:

```text
No Risk
Risk
```

### 5. RoBERTa Multiclass Classifier

A fine-tuned RoBERTa model performs four-class classification:

```text
High Risk
Medium Risk
Low Risk
No Risk
```

## Data Processing

The preprocessing pipeline includes:

- Unicode normalization
- HTML removal
- URL removal
- Special-character handling
- Repeated-symbol normalization
- Whitespace normalization
- Emoji normalization
- Text standardization
- Dataset merging
- Label encoding

The dissertation also analyzes non-standard symbols and emoji distributions before and after preprocessing.

## Feature Engineering

### MiniLM Embeddings

Cleaned text is converted into 384-dimensional sentence embeddings using a pretrained MiniLM sentence-transformer model.

### TF-IDF

TF-IDF is used as a classical text representation for baseline models.

### Risk Lexicon Features

A custom lexicon-based feature transformer is incorporated to capture risk-related terms in the input text.

## Evaluation

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC/AUC for binary classification
- Training and validation loss
- Robustness under controlled noise

According to the dissertation results:

| Model | Reported Result |
|---|---:|
| LightGBM Multiclass | ~96% test accuracy |
| Logistic Regression Multiclass | ~91–92% test accuracy |
| RoBERTa Multiclass | 81% test accuracy |
| RoBERTa Binary | ~91% test accuracy |
| RoBERTa Binary | AUC ≈ 0.92 |

The reported RoBERTa multiclass results include approximately 0.81 weighted F1-score, while the binary model reports approximately 0.91 weighted F1-score.

## Explainability

Integrated Gradients was used to analyze which tokens contribute to model predictions.

The dissertation reports that the analysis highlighted:

- Emotional expressions
- Self-harm-related terms
- Negative phrases
- Anxiety-related expressions

This provides an additional layer of interpretability for the transformer-based predictions.

## Robustness Study

The RoBERTa model was evaluated after introducing controlled text noise.

The dissertation reports that noise injection produced approximately:

- 0.80 test accuracy
- Lower confidence
- Better separation between risk classes in some noisy examples

This experiment was used to examine whether the model could generalize beyond exact memorization of training examples.

## Paragraph-Level Prediction

Longer inputs are divided into manageable text chunks before classification.

Predictions from individual chunks are then aggregated using a voting-based approach to produce a paragraph-level result.

This allows the system to process longer text such as:

- Journal-style writing
- Long-form posts
- Extended self-expression
- Multi-sentence user input

## Project Structure

A typical repository structure can be organized as:

```text
.
├── README.md
├── data/
├── notebooks/
├── models/
├── src/
├── embeddings/
├── results/
├── requirements.txt
└── app.py
```

The exact files and directories may vary depending on the final repository version.

## Main Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- LightGBM
- PyTorch
- Transformers
- Sentence-Transformers
- RoBERTa
- MiniLM
- SciPy
- Matplotlib
- Google Colab

## Limitations

The dissertation identifies several limitations:

- Ambiguity between neutral, insulting, and hate-speech-like language can affect classification.
- Sarcasm, multilingual text, slang, and code-mixed language remain challenging.
- Context-dependent meaning can be difficult to infer from isolated sentences.
- Some risk categories have limited or imbalanced examples.
- Transformer embeddings may discard some contextual information when inputs are aggressively shortened.
- Automated classification should not be treated as a substitute for professional assessment or human review.

## Results and Discussion

The dissertation reports that preprocessing improved the quality of the text data, while transformer-based embeddings and boosted-tree models provided useful representations for classification.

The project also compares classical ML approaches with fine-tuned transformer models and examines robustness and explainability.

The final system is designed as a real-time text classification pipeline that can process user input and return a risk-related prediction.

## Dissertation

This repository is associated with the dissertation:

**Multi-Stage Suicide Risk Detection via Cognitive & Topic-Aware Models**

Master of Science in Data Science  
Vellore Institute of Technology, Chennai

## Author

**Sinthesh S**

## Disclaimer

This project is an academic/research prototype for text classification. Predictions from the system should not be interpreted as medical or clinical diagnoses, nor should they be used as the sole basis for decisions concerning an individual's safety or mental health.
