from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import os
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for React app

# Load the trained model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'public', 'models', 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'public', 'models', 'vectorizer.pkl')

print("Loading model and vectorizer...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
print("Model loaded successfully!")

def preprocess_text(text):
    """Preprocess text similar to training preprocessing"""
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    text = ' '.join(text.split())
    return text

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'Logistic Regression',
        'version': '1.0'
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict if an email is spam or not"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        email_text = data['text']
        if not email_text or len(email_text.strip()) == 0:
            return jsonify({'error': 'Email text cannot be empty'}), 400
        
        # Preprocess the text
        processed_text = preprocess_text(email_text)
        
        # Transform using the vectorizer
        text_vectorized = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        
        # Get prediction probability
        try:
            probability = model.predict_proba(text_vectorized)[0]
            confidence = float(max(probability))
            spam_probability = float(probability[1]) if len(probability) > 1 else (1.0 if prediction == 1 else 0.0)
        except AttributeError:
            confidence = 1.0
            spam_probability = 1.0 if prediction == 1 else 0.0
        
        result = {
            'isSpam': bool(prediction == 1),
            'confidence': confidence,
            'spamProbability': spam_probability,
            'prediction': 'spam' if prediction == 1 else 'legitimate'
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model metadata"""
    metadata_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'models', 'model_metadata.json')
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': f'Could not load metadata: {str(e)}'}), 500

# Optional: Add root route to avoid 404 at /
@app.route('/')
def home():
    return "Flask backend is running!"

if __name__ == '__main__':
    print("Starting Flask API server...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Vectorizer path: {VECTORIZER_PATH}")
    app.run(debug=True, host='0.0.0.0', port=5000)
