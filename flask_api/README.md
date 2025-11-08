# Flask API for Spam Detection Model

This Flask API serves your trained Logistic Regression spam detection model.

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   cd flask_api
   pip install -r requirements.txt
   ```

2. **Run the Flask server:**
   ```bash
   python app.py
   ```
   
   The API will start on `http://localhost:5000`

## API Endpoints

### POST /predict
Predict if an email is spam or not.

**Request:**
```json
{
  "text": "Your email text here"
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
Health check endpoint.

### GET /model-info
Get model metadata including F1 score, accuracy, etc.

## Testing

You can test the API using curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won $1000. Click here now!"}'
```

## Deployment Options

- **Local Development:** Run locally as described above
- **Heroku:** Use `Procfile` for deployment
- **AWS Lambda:** Use Zappa or similar
- **Google Cloud Run:** Containerize and deploy
- **DigitalOcean App Platform:** Direct deployment

## Model Information

- **Model:** Logistic Regression
- **F1 Score:** 0.968
- **Accuracy:** 98.2%
- **Features:** 3000 TF-IDF features
