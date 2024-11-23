import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

class FraudDetectionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None

    def load_and_preprocess(self, train_path, test_path=None):
        # Load and preprocess the dataset
        print("Loading and preprocessing data...")
        
        # Load training data
        train_data = pd.read_csv(train_path)
        
        # Separate features and target
        if 'Class' in train_data.columns:
            self.target_column = 'Class'
        else:
            # Try to identify the target column based on binary values
            binary_cols = train_data.select_dtypes(include=[np.number]).columns[
                train_data.select_dtypes(include=[np.number]).apply(lambda x: set(x.unique()) == {0, 1})
            ]
            if len(binary_cols) == 1:
                self.target_column = binary_cols[0]
            else:
                raise ValueError("Could not identify target column")

        # Remove any non-numeric columns
        numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != self.target_column]
        self.feature_columns = numeric_columns

        # Split features and target
        X_train = train_data[self.feature_columns]
        y_train = train_data[self.target_column]

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)

        # Handle imbalanced data using SMOTE
        print("Applying SMOTE to handle imbalanced data...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        if test_path:
            # Load and preprocess test data
            test_data = pd.read_csv(test_path)
            X_test = test_data[self.feature_columns]
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
            return X_train_balanced, y_train_balanced, X_test_scaled
        
        return X_train_balanced, y_train_balanced

    def train_and_evaluate(self, X_train, y_train):
        # Train and evaluate multiple models
        print("\nTraining and evaluating models...")
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }

        best_auc = 0
        results = {}

        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            mean_cv_score = cv_scores.mean()
            
            # Train model on full training data
            model.fit(X_train, y_train)
            
            # Store results
            results[name] = {
                'model': model,
                'cv_score': mean_cv_score
            }
            
            print(f"{name} CV ROC-AUC Score: {mean_cv_score:.4f}")
            
            # Update best model if current is better
            if mean_cv_score > best_auc:
                best_auc = mean_cv_score
                self.best_model = model
                self.best_model_name = name

        print(f"\nBest Model: {self.best_model_name} with ROC-AUC: {best_auc:.4f}")
        return results

    def predict(self, X_test):
        # Make predictions using the best model
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.best_model.predict(X_test)

    def save_predictions(self, X_test, predictions, output_path):
        # Save predictions to a CSV file
        output_df = pd.DataFrame()
        output_df['Transaction_ID'] = range(len(predictions))
        output_df['Predicted_Class'] = predictions
        output_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")

def main():
    # Initialize the model
    model = FraudDetectionModel()

    # Load and preprocess data
    X_train, y_train, X_test = model.load_and_preprocess(
        train_path='fraudTrain.csv',
        test_path='fraudTest.csv'
    )

    # Train and evaluate models
    results = model.train_and_evaluate(X_train, y_train)

    # Make predictions on test data
    predictions = model.predict(X_test)

    # Save predictions
    model.save_predictions(X_test, predictions, 'fraud_predictions.csv')

if __name__ == "__main__":
    main()