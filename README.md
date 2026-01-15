# 🖥️ Aspect-Based Sentiment Analysis for Laptop Reviews  
### Two-Stage BERT + CRF Pipeline

## 📌 One-Line Description
A transformer-based Aspect-Based Sentiment Analysis system that extracts laptop aspects using **BERT+CRF** and classifies sentiment using **BERT** for fine-grained review analysis.

---

## 📖 Project Overview
Traditional sentiment analysis assigns a single sentiment to an entire review, which often misses contrasting opinions about different product features.  
This project implements a **two-stage Aspect-Based Sentiment Analysis (ABSA) pipeline** that:

1. Extracts explicit aspect terms (e.g., *battery life, display, keyboard*)
2. Classifies sentiment for each extracted aspect as **Positive**, **Negative**, or **Neutral**

The system is designed for **laptop product reviews** with a focus on **accuracy, interpretability, and modularity**.

---

## 🧠 Model Architecture

### 🔹 Stage 1: Aspect Term Extraction (ATE)
- Model: **BERT + Conditional Random Field (CRF)**
- Task: Token-level BIO tagging
- Purpose: Accurate extraction of multi-word aspect terms

### 🔹 Stage 2: Aspect Sentiment Classification (ASC)
- Model: **BERT Sequence Classification**
- Task: Aspect-level sentiment classification
- Classes: Positive | Negative | Neutral

---

## 📊 Results

### Aspect Term Extraction
- Exact F1-score: **0.8129**
- Overlap F1-score: **0.9102**

### Aspect Sentiment Classification
- Validation Accuracy: **77.54%**
- Macro F1-score: **0.7437**

### End-to-End Pipeline
- Precision: **0.9216**
- Recall: **0.9592**
- F1-score: **0.9400**

---

## 🛠️ Tech Stack
- Python  
- PyTorch  
- Hugging Face Transformers  
- BERT (bert-base-uncased)  
- Conditional Random Fields (CRF)  
- NLTK / spaCy  
- BART & PEGASUS (Summarization)

---

## 📂 Project Structure
```bash
├── data/
├── aspect_extraction/
├── sentiment_classification/
├── pipeline/
├── outputs/
├── NLP_SUBMISSION.ipynb
├── README.md
```

---

## 🚀 How to Run

### Install Dependencies
```bash
pip install torch transformers scikit-learn pandas numpy
```

### Train Models
```bash
python aspect_extraction/train_ate.py
python sentiment_classification/train_asc.py
```

### Run Full Pipeline
```bash
python pipeline/run_pipeline.py
```

---

## ✨ Key Features
- Fine-grained aspect-level sentiment analysis  
- CRF-based structured decoding  
- Modular and interpretable architecture  
- Generates aspect–sentiment insights and summaries  
- Easily extendable to other domains  

---

## 🔮 Future Scope
- Implicit aspect detection  
- Cross-domain and multilingual ABSA  
- Model optimization for deployment  
- Multimodal sentiment analysis  

