import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class SMSSpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000),
            'svm': LinearSVC(random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.vectorizer_fitted = False
        
    def clean_text(self, text):
        """
        Special cleaning for SMS text data
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3,}\b', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s,.!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess_text(self, text):
        # Initial cleaning
        text = self.clean_text(text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Join tokens back into text
        return ' '.join(tokens)

    def load_and_preprocess_data(self, file_path):
        try:
            # Load the dataset with specific format
            df = pd.read_csv(file_path, encoding='latin-1')
            
            # Select only the first two columns (v1 and v2)
            df = df.iloc[:, 0:2]
            
            # Rename columns for clarity
            df.columns = ['label', 'message']
            
            # Convert labels to binary (spam = 1, ham = 0)
            df['label'] = (df['label'] == 'spam').astype(int)
            
            # Convert messages to string type and handle missing values
            X = df['message'].fillna('').astype(str)
            y = df['label']

            # Preprocess all messages
            print("Preprocessing messages...")
            X = X.apply(self.preprocess_text)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            print(f"Training set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}")
            print(f"Spam messages in training set: {sum(y_train == 1)}")
            print(f"Ham messages in training set: {sum(y_train == 0)}")

            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def train_models(self, X_train, X_test, y_train, y_test):
        # Convert text to TF-IDF features
        print("Converting text to TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        self.vectorizer_fitted = True

        best_accuracy = 0
        results = {}

        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            results[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred
            }
            
            # Keep track of the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name

        return results

    def predict(self, text):
        if not self.vectorizer_fitted or self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        
        # Transform text to TF-IDF features
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.best_model.predict(text_tfidf)[0]
        
        # Get prediction probability if the model supports it
        probability = None
        if hasattr(self.best_model, 'predict_proba'):
            probability = self.best_model.predict_proba(text_tfidf)[0]
        
        return {
            'text': text,
            'processed_text': processed_text,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': probability[1] if probability is not None else None,
            'model_used': self.best_model_name
        }

# Example usage
def main():
    # Initialize classifier
    classifier = SMSSpamClassifier()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_split = classifier.load_and_preprocess_data('spam.csv')
    
    if data_split is not None:
        X_train, X_test, y_train, y_test = data_split
        
        # Train models
        print("Training models...")
        results = classifier.train_models(X_train, X_test, y_train, y_test)
        
        # Print results
        for model_name, metrics in results.items():
            print(f"\nResults for {model_name}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print("\nClassification Report:")
            print(metrics['classification_report'])
            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
        
        # Example predictions
        example_texts = [
            "Go until jurong point, crazy.. Available only in bugis n great world la e buffet...",
            "URGENT! Winner! You have won a 1 week FREE membership in our prize draw! Text WIN to 87070.",
            "Ok lar... Joking wif u oni..."
        ]
        
        print("\nExample predictions:")
        for text in example_texts:
            result = classifier.predict(text)
            print(f"\nText: {result['text']}")
            print(f"Processed text: {result['processed_text']}")
            print(f"Prediction: {result['prediction']}")
            if result.get('confidence') is not None:
                print(f"Confidence: {result['confidence']:.4f}")
            print(f"Model used: {result['model_used']}")

if __name__ == "__main__":
    main()