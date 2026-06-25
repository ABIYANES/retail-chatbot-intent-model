# Retail E-commerce Chatbot Intent Classifier

This repository contains a structured machine learning pipeline built to classify customer intents based on e-commerce retail queries using the `bitext-retail-ecommerce-llm-chatbot-training-dataset.csv` dataset.

## 📁 Project Structure
* `data/` - Contains the raw retail e-commerce dataset.
* `src/` - Production-ready Python source code (including `train.py`).
* `models/` - Saved model artifacts and vectorizers (ignored by Git).
* `notebooks/` - Jupyter notebooks for data exploration.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/ABIYANES/retail-chatbot-intent-model.git](https://github.com/ABIYANES/retail-chatbot-intent-model.git)
cd retail-chatbot
python -m venv venv
source venv/Scripts/activate  # On Mac/Linux use: source venv/bin/activate
pip install -r requirements.txt