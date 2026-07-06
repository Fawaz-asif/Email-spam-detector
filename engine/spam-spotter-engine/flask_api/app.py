from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import os
import json
from typing import Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    resources = ["stopwords", "wordnet", "omw-1.4"]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK resource {res}: {e}")

download_nltk_resources()

app = Flask(__name__)
CORS(app, origins=os.environ.get('CORS_ORIGINS', '*').split(','))

# Load the trained model and vectorizer
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public', 'models'))
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')

SPAM_CLASS_LABEL = 1

# Setup stopwords and lemmatizer exactly like in training pipeline
STOP_WORDS = set(stopwords.words("english"))
for keep in ("not", "no"):
    STOP_WORDS.discard(keep)

lemmatizer = WordNetLemmatizer()


def load_pickle(path: str, artifact_name: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f'{artifact_name} not found at {path}')

    with open(path, 'rb') as f:
        return pickle.load(f)


def load_metadata() -> dict[str, Any]:
    if not os.path.exists(METADATA_PATH):
        return {}

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


print("Loading model and vectorizer...")
model = load_pickle(MODEL_PATH, 'Model')
vectorizer = load_pickle(VECTORIZER_PATH, 'Vectorizer')
metadata = load_metadata()
print("Model loaded successfully!")

def preprocess_text(text):
    """Preprocess text exactly matching the training pipeline before TF-IDF."""
    try:
        if not isinstance(text, str):
            text = str(text)

        # Lowercase
        text = text.lower()

        # Remove non-letters (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize using regex to extract alphabetic tokens
        tokens = re.findall(r'\b[a-z]+\b', text)

        # Stopword removal
        filtered = [t for t in tokens if t not in STOP_WORDS]

        # Lemmatize
        lemmatized = [lemmatizer.lemmatize(t) for t in filtered]

        return ' '.join(lemmatized)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': metadata.get('model_name', 'unknown'),
        'featureCount': metadata.get('feature_count'),
        'version': '1.0',
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict if an email is spam or not"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json(silent=True)
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        email_text = data['text']
        if not isinstance(email_text, str):
            return jsonify({'error': 'Email text must be a string'}), 400

        if not email_text or len(email_text.strip()) == 0:
            return jsonify({'error': 'Email text cannot be empty'}), 400
        
        # Preprocess the text
        processed_text = preprocess_text(email_text)
        
        # Transform using the vectorizer
        text_vectorized = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = int(model.predict(text_vectorized)[0])
        
        # Get prediction probability
        try:
            probability = model.predict_proba(text_vectorized)[0]
            confidence = float(max(probability))
            classes = list(getattr(model, 'classes_', []))
            spam_index = classes.index(SPAM_CLASS_LABEL) if SPAM_CLASS_LABEL in classes else None
            spam_probability = float(probability[spam_index]) if spam_index is not None else (1.0 if prediction == SPAM_CLASS_LABEL else 0.0)
        except AttributeError:
            confidence = 1.0
            spam_probability = 1.0 if prediction == SPAM_CLASS_LABEL else 0.0
        
        result = {
            'isSpam': bool(prediction == SPAM_CLASS_LABEL),
            'confidence': confidence,
            'spamProbability': spam_probability,
            'prediction': 'spam' if prediction == SPAM_CLASS_LABEL else 'legitimate'
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model metadata"""
    if metadata:
        return jsonify(metadata)

    return jsonify({'error': 'Model metadata not found'}), 404

# Optional: Add root route to avoid 404 at /
@app.route('/')
def home():
    return jsonify({
        'message': 'Flask backend is running',
        'endpoints': ['/health', '/predict', '/model-info'],
    })

if __name__ == '__main__':
    print("Starting Flask API server...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Vectorizer path: {VECTORIZER_PATH}")
    app.run(
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true',
        host=os.environ.get('FLASK_HOST', '0.0.0.0'),
        port=int(os.environ.get('FLASK_PORT', '5000')),
    )
