# Email Spam Detector Setup Guide

Your trained ML model is now integrated! Follow these steps to run the full application.

## Quick Start

### 1. Start the Flask API (Backend)

```bash
# Navigate to the Flask API directory
cd flask_api

# Install Python dependencies
pip install -r requirements.txt

# Run the Flask server
python app.py
```

The API will start on `http://localhost:5000`

### 2. Start the React App (Frontend)

In a **new terminal**:

```bash
# The React app should already be running in Lovable
# If not, it will start automatically
```

The React app connects to your Flask API and uses your trained model for predictions!

## How It Works

1. **Your Model Files** (in `public/models/`):
   - `model.pkl` - Your trained Logistic Regression model
   - `vectorizer.pkl` - TF-IDF vectorizer with 3000 features
   - `model_metadata.json` - Model performance metrics

2. **Flask API** (`flask_api/app.py`):
   - Loads your trained model
   - Provides `/predict` endpoint for spam detection
   - Preprocesses text the same way as during training
   - Returns predictions with confidence scores

3. **React Frontend** (`src/components/SpamDetector.tsx`):
   - Sends email text to Flask API
   - Displays predictions and confidence scores
   - Shows real-time API connection status

## API Endpoints

### POST /predict
Analyze email text for spam.

**Request:**
```json
{
  "text": "Your email content here..."
}
```

**Response:**
```json
{
  "isSpam": true,
  "confidence": 0.95,
  "spamProbability": 0.95,
  "prediction": "spam"
}
```

### GET /health
Check if API is running.

### GET /model-info
Get model metadata (accuracy, F1 score, etc.)

## Troubleshooting

### API Connection Error
- ✅ Make sure Flask is running: `python flask_api/app.py`
- ✅ Check the console for any errors
- ✅ Verify the API is accessible at `http://localhost:5000`

### Model Loading Error
- ✅ Ensure `model.pkl` and `vectorizer.pkl` are in `public/models/`
- ✅ Check scikit-learn version compatibility

### CORS Error
- ✅ Flask-CORS is already configured in `app.py`
- ✅ Make sure you installed dependencies: `pip install flask-cors`

## Deployment Options

When ready to deploy:

1. **Flask API**: Deploy to Heroku, AWS Lambda, Google Cloud Run, or DigitalOcean
2. **Update API_URL**: Change the `API_URL` in `src/components/SpamDetector.tsx` to your deployed API URL
3. **React App**: Already handled by Lovable!

## Model Performance

Your trained model:
- **Algorithm**: Logistic Regression
- **F1 Score**: 96.8%
- **Accuracy**: 98.2%
- **Precision**: 96.4%
- **Recall**: 97.3%
- **Features**: 3000 TF-IDF features

Excellent performance! 🎉
