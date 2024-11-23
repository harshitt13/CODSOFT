import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_data(filepath, is_training=True):
    # Load data from a custom-formatted file
    try:
        # Read file with explicit encoding
        df = pd.read_csv(filepath, sep=' ::: ', engine='python', header=None)
        
        if is_training:
            # Training data format: ID ::: TITLE ::: GENRE ::: DESCRIPTION
            df.columns = ['id', 'title', 'genre', 'description']
            # Remove any rows with missing genres
            df = df.dropna(subset=['genre'])
            
            # Print genre distribution
            print("Genre Distribution:")
            print(df['genre'].value_counts())
        else:
            # Test data format: ID ::: TITLE ::: DESCRIPTION
            df.columns = ['id', 'title', 'description']
        
        # Check if dataframe is empty
        if df.empty:
            print(f"Error: No valid data found in {filepath}")
            return None
        
        # For training data, check unique genres
        if is_training:
            unique_genres = df['genre'].nunique()
            if unique_genres < 2:
                print(f"Error: Need at least 2 different genres. Found {unique_genres}")
                return None
        
        return df
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def train_genre_predictor(train_filepath):
    # Train movie genre prediction model
    # Load training data
    data = load_data(train_filepath, is_training=True)
    if data is None:
        return None

    # Encode genres
    le = LabelEncoder()
    data['genre_encoded'] = le.fit_transform(data['genre'])

    # Print label mapping
    print("\nGenre Label Mapping:")
    for genre, label in zip(le.classes_, le.transform(le.classes_)):
        print(f"{genre}: {label}")

    # Create pipeline with TF-IDF vectorization and LinearSVC
    genre_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('classifier', LinearSVC(random_state=42))
    ])

    # Print some diagnostic information
    print(f"\nTraining data shape: {data.shape}")
    print("Sample descriptions:")
    print(data['description'].head())

    try:
        # Train on full dataset
        genre_pipeline.fit(data['description'], data['genre_encoded'])
    except ValueError as e:
        print(f"Training error: {e}")
        return None

    return {
        'pipeline': genre_pipeline,
        'label_encoder': le
    }

def test_and_save_predictions(model, test_filepath, output_filepath):
    """Test model and save predictions"""
    if model is None:
        print("Model not trained.")
        return

    # Load test data
    test_data = load_data(test_filepath, is_training=False)
    if test_data is None:
        return

    # Print test data information
    print(f"\nTest data shape: {test_data.shape}")
    print("Sample test descriptions:")
    print(test_data['description'].head())

    # Predict genres
    predictions = model['pipeline'].predict(test_data['description'])
    predicted_genres = model['label_encoder'].inverse_transform(predictions)

    # Prepare output data
    output_data = []
    for (id_val, title), predicted_genre in zip(zip(test_data['id'], test_data['title']), predicted_genres):
        output_data.append(f"{id_val} ::: {title} ::: {predicted_genre}")

    # Save predictions
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_data))
    
    print(f"Predictions saved to {output_filepath}")

# Main execution
if __name__ == "__main__":
    # Train the model
    trained_model = train_genre_predictor('train_data.txt')

    # Test and save predictions
    if trained_model:
        test_and_save_predictions(
            trained_model, 
            'test_data.txt', 
            'classified_genre.txt'
        )