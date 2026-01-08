# Spam Message Detection System (NLP + Machine Learning)

An end-to-end spam message detection system built using classical Machine Learning and Natural Language Processing, enhanced with a hybrid rule-based + ML approach to handle real-world adversarial spam.

## Project Overview

### Live Demo
- Streamlit App: https://spam-message-detection-ml.streamlit.app/
Spam messages often rely on:

- **Obfuscated text** (e.g., fr33, m0n3y)
- **Reassuring phrases** (e.g., "this is not a scam")
- **Social-engineering techniques**

Pure ML models struggle with such cases.

This project addresses that limitation by combining:

TF-IDF + Logistic Regression (ML)

- **TF-IDF + Logistic Regression** (ML)
- **Rule-based detection** (explicit human logic)

The final system is deployed as a **Streamlit web application**.

## Key Features

- ✓ Classical NLP (no deep learning)
- ✓ Handles obfuscated spam text
- ✓ Hybrid rule-based and ML detection
- ✓ High precision and recall
- ✓ Real-time predictions via Streamlit
- ✓ ## System Architecture
User Message
   ↓
Character Normalization
   ↓
Rule-Based Detection (Reassurance Phrases)
   ↓
Text Cleaning + Stemming
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression Classifier
   ↓
Spam / Not Spam Prediction

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| NLP | NLTK |
| Machine Learning | Scikit-learn |
| Vectorization | TF-IDF |
| Model | Logistic Regression |
| Deployment | Streamlit |
| Model Storage | Joblib |

## Project Structure

```
spam-detection/
│
├── trained_models/
│   ├── tfidf_vectorizer.pkl
│   └── spam_classifier.pkl
│
├── dataset/
│   └── spam.csv
│
├── spam_detection.ipynb
├── app.py
├── Readme.md
└── requirements.txt
```

## Model Performance

| Metric | Ham     | Spam |
|--------|---------|------|
| Precision | 0.99    | 0.97 |
| Recall | 1.00    | 0.93 |
| F1-score | 0.99    | 0.95 |
| **Accuracy** | **99%** | -    |

> Evaluation focuses on precision, recall, and F1-score due to class imbalance.

## Why Hybrid Detection Is Required

### Limitation of ML-only Models

Messages such as:

> "This is not a scam. You have been chosen for a reward"

Often bypass ML classifiers because reassuring language strongly resembles legitimate messages.

### Solution

A rule-based override layer detects known scam tactics before ML prediction. This approach reflects how real-world spam filters are designed.

## Example Test Cases

### ✗ Spam
- "WIN a free iPhone now! Click here"
- "C0ngr@tulat10ns! Y0u w0n fr33 m0n3y"
- "This is not a scam. You have been chosen"

### ✓ Not Spam
- "Hey, are we meeting today?"
- "Please submit the assignment by tomorrow"

## How to Run the Application

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit app

```bash
streamlit run app.py
```

### Step 3: Open in browser

Navigate to: http://localhost:8501
