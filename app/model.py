import json
import math
from collections import defaultdict
from features import extract_features

DB_FILE = "database.json"
MODEL_FILE = "model.json"
ALPHA = 1  # Laplace smoothing

def load_db():
    """Load the database from JSON, handling missing file."""
    try:
        with open(DB_FILE, encoding="utf-8-sig") as f:
            data = json.load(f)
            # Convert dicts back to defaultdict
            data["fake"] = defaultdict(int, data.get("fake", {}))
            data["real"] = defaultdict(int, data.get("real", {}))
            data["fake_docs"] = data.get("fake_docs", 0)
            data["real_docs"] = data.get("real_docs", 0)
            return data
    except FileNotFoundError:
        return {"fake": defaultdict(int), "real": defaultdict(int), "fake_docs": 0, "real_docs": 0}

def save_db(db):
    """Save database to JSON."""
    db_copy = db.copy()
    db_copy["fake"] = dict(db_copy["fake"])
    db_copy["real"] = dict(db_copy["real"])
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db_copy, f, indent=2)

def compute_model():
    """Compute probabilities for all features and priors."""
    db = load_db()
    fake_counts = db["fake"]
    real_counts = db["real"]

    all_features = set(fake_counts.keys()) | set(real_counts.keys())

    total_fake = sum(fake_counts.values())
    total_real = sum(real_counts.values())
    feature_probs = {"Fake": {}, "Real": {}}

    for f in all_features:
        feature_probs["Fake"][f] = (fake_counts.get(f, 0) + ALPHA) / (total_fake + ALPHA * len(all_features))
        feature_probs["Real"][f] = (real_counts.get(f, 0) + ALPHA) / (total_real + ALPHA * len(all_features))

    # Smoothed priors
    total_docs = db["fake_docs"] + db["real_docs"]
    P_Fake = (db["fake_docs"] + ALPHA) / (total_docs + 2 * ALPHA)
    P_Real = (db["real_docs"] + ALPHA) / (total_docs + 2 * ALPHA)

    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "feature_probs": feature_probs,
            "P_Fake": P_Fake,
            "P_Real": P_Real
        }, f, indent=2)

def predict(text, url="", model=None):
    """Predict if a text is fake or real."""
    if model is None:
        with open(MODEL_FILE, encoding="utf-8-sig") as f:
            model = json.load(f)

    feats = extract_features(text, url)
    probs = model["feature_probs"]
    P_Fake = model["P_Fake"]
    P_Real = model["P_Real"]

    # Safe log probabilities
    fake_log_sum = math.log(P_Fake)
    real_log_sum = math.log(P_Real)

    # Total counts for smoothing unseen features
    total_fake_counts = sum(model["feature_probs"]["Fake"].values())
    total_real_counts = sum(model["feature_probs"]["Real"].values())
    n_fake_features = len(model["feature_probs"]["Fake"])
    n_real_features = len(model["feature_probs"]["Real"])

    for f, v in feats.items():
        if v > 0:
            p_fake = probs["Fake"].get(f, ALPHA / (total_fake_counts + ALPHA * n_fake_features))
            p_real = probs["Real"].get(f, ALPHA / (total_real_counts + ALPHA * n_real_features))
            fake_log_sum += v * math.log(p_fake)
            real_log_sum += v * math.log(p_real)

    # Convert log probabilities back to normal probability
    max_log = max(fake_log_sum, real_log_sum)
    fake_exp = math.exp(fake_log_sum - max_log)
    real_exp = math.exp(real_log_sum - max_log)
    prob_fake = fake_exp / (fake_exp + real_exp)

    return {
        "verdict": "fake" if prob_fake > 0.5 else "real",
        "probability": prob_fake,
        "features": feats
    }

def update_model(text, url, label):
    """Update the database and recompute the model based on user feedback."""
    db = load_db()
    feats = extract_features(text, url)

    if label.lower() == "fake":
        db["fake_docs"] += 1
        for k, v in feats.items():
            db["fake"][k] = db["fake"].get(k, 0) + v
    else:
        db["real_docs"] += 1
        for k, v in feats.items():
            db["real"][k] = db["real"].get(k, 0) + v

    save_db(db)
    compute_model()

# Initialize model on first run
if __name__ == "__main__":
    compute_model()
