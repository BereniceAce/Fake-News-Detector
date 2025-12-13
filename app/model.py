import math
from features import extract_features
import json

MODEL_FILE = "model.json"

with open(MODEL_FILE, encoding="utf-8-sig") as f:
    MODEL = json.load(f)

def predict(text):
    feats = extract_features(text)

    # Priors for Bayes
    P_fake = MODEL["P_Fake"]
    P_real = MODEL["P_Real"]

    log_fake = math.log(P_fake)
    log_real = math.log(P_real)

    # Iterate over list of features needed for computation
    for f in feats:
        log_fake += math.log(MODEL["feature_probs"]["Fake"].get(f, 1e-6))
        log_real += math.log(MODEL["feature_probs"]["Real"].get(f, 1e-6))

    # Softmax normalization
    max_log = max(log_fake, log_real)
    fake_exp = math.exp(log_fake - max_log)
    real_exp = math.exp(log_real - max_log)
    prob_fake = fake_exp / (fake_exp + real_exp)

    return {
        "verdict": "Fake" if prob_fake > 0.5 else "Real",
        "probability": round(prob_fake * 100, 2)
    }
