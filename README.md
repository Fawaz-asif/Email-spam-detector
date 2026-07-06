# Email Spam Detector

An end-to-end Machine Learning-powered Email Spam Detector that uses a trained Logistic Regression model to classify emails as spam or legitimate. 

The project includes:
1. **Machine Learning Pipeline (`/pipeline`)**: Scripts to preprocess data, perform Exploratory Data Analysis (EDA), train multiple classifier models (Naive Bayes, Logistic Regression, Support Vector Machines, Random Forest, and Gradient Boosting), compare their performance, and select the best model.
2. **Web Application Engine (`/engine/spam-spotter-engine`)**: A web interface consisting of:
   - **Flask API (`/flask_api`)**: Serving the trained model using matching preprocessing steps.
   - **Vite React Frontend (`/frontend`)**: A modern, premium UI built with React, TypeScript, Tailwind CSS, and shadcn/ui.

---

## Folder Structure

```
Email spam detector/
├── .gitignore
├── README.md                 # Root documentation (this file)
├── pipeline/                 # Machine Learning training pipeline
│   ├── spam.py               # Main training script (runs EDA, preprocesses, trains & syncs)
│   ├── requirements.txt      # Python dependencies for pipeline
│   ├── archive/              # Raw datasets
│   │   └── spam_ham_dataset.csv
│   └── eda_plots/            # Generated EDA visualization plots
└── engine/
    └── spam-spotter-engine/
        ├── flask_api/        # Flask backend API
        │   ├── app.py        # Backend server entry point
        │   ├── requirements.txt # Python dependencies for API
        │   └── venv/         # Virtual environment
        └── frontend/         # React/Vite Frontend
            ├── package.json  # NPM scripts and dependencies
            ├── tailwind.config.ts
            ├── src/
            │   ├── components/ # React components (SpamDetector.tsx)
            │   └── pages/      # Pages (Index.tsx, NotFound.tsx)
            └── public/
                └── models/   # Loaded ML models (Synced from pipeline)
                    ├── model.pkl
                    ├── vectorizer.pkl
                    └── model_metadata.json
```

---

## Setup & Running Guide

### 1. Training & Syncing the ML Model

The training script automatically performs EDA, tokenizes, removes stopwords, lemmatizes with WordNet, trains/evaluates 5 different models, and serializes the best one. Upon success, it automatically synchronizes the model files to the web app's models directory (`/engine/spam-spotter-engine/frontend/public/models`).

To train the model:
```bash
# Navigate to the pipeline directory
cd pipeline

# Install requirements
pip install -r requirements.txt

# Run the pipeline (this will auto-sync model artifacts to the engine)
python spam.py
```

*Note: You can skip the long preprocessing phase using `python spam.py --phase select` if preprocessed CSVs (`train_data.csv` and `test_data.csv`) already exist.*

### 2. Setting Up the Web Application

The project comes configured to run the backend and frontend concurrently.

#### Prerequisites
- Node.js & npm (v18+)
- Python 3.10+

#### Installation

1. **Install Frontend Dependencies:**
   ```bash
   cd engine/spam-spotter-engine/frontend
   npm install
   ```

2. **Configure and Install Backend dependencies:**
   The backend dependencies should be installed in a virtual environment under the `flask_api` folder:
   ```bash
   cd ../flask_api
   python -m venv venv
   .\venv\Scripts\pip install -r requirements.txt  # Windows
   # OR: source venv/bin/activate && pip install -r requirements.txt  # Linux/Mac
   ```

#### Running the Development Servers

From the `engine/spam-spotter-engine/frontend` directory, start both the Flask API backend and Vite React frontend with a single command:
```bash
cd engine/spam-spotter-engine/frontend
npm run dev
```

- The React frontend will start on: `http://localhost:8080`
- The Flask API backend will start on: `http://localhost:5000`

---

## Machine Learning Details

- **Features Extracted:** TF-IDF (Term Frequency-Inverse Document Frequency) using unigrams and bigrams (max 3,000 features).
- **Text Preprocessing (Identical in training & API):**
  - Case folding (lowercasing).
  - Punctuation and non-alphabetic characters removal.
  - Custom stopword removal (preserving negation terms `no` and `not`).
  - NLTK WordNet Lemmatization.
- **Model Performance:**
  - **Selected Model:** Logistic Regression
  - **F1 Score:** ~96.8%
  - **Accuracy:** ~98.2%
