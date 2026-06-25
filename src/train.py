import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

def train_model():
    # 1. Load data
    data_path = os.path.join('data', 'bitext-retail-ecommerce-llm-chatbot-training-dataset.csv')
    df = pd.read_csv(data_path)
    
    # 2. Define features and target
    X = df['instruction']
    y = df['intent']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Feature Extraction
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Model Training
    print("Training the Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # 6. Evaluation
    y_pred = model.predict(X_test_vec)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save Artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/intent_classifier.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Model and vectorizer saved successfully in /models!")

if __name__ == "__main__":
    train_model()