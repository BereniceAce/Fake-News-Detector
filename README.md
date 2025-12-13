# Fake News Detector

This project implements a **Naive Bayes-based fake news detector** using feature-based analysis. It predicts whether a piece of text is **Fake** or **Real** based on linguistic, academic, and technical features.

---

## Features

- Detects suspicious punctuation, spelling issues, and fake keywords.
- Analyzes academic and technical vocabulary levels.
- Uses a precomputed JSON model of feature probabilities (`model.json`).
- Built with **Python 3** and **Flask** for a simple web interface.
- Lightweight and easy to run locally.

---

## File Structure

Fake-News-Detector/
│
├─ flask_api.py # Flask API for serving predictions
├─ model.py # Naive Bayes prediction logic
├─ features.py # Feature extraction logic
├─ model.json # Pre-trained feature probability model
├─ templates/
│ └─ frontend.html # Simple HTML frontend
├─ static/
│ └─ frontend.css # Styling for frontend
└─ README.md

---

## Prerequisites

- Python 3.10+ installed
- `pip` package manager
- Recommended: create a virtual environment

bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install required
pip install flask


## Running Locally
1. Start the API
python flask_api.py
2. Open browser Visit: http://127.0.0.1:5000

You will see a simple form to enter text for analysis.
Send text for prediction
The frontend sends a POST request to /predict.
The API returns a JSON response:

{
  "verdict": "Fake",
  "probability": 87.65
}


