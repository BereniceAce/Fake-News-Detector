import pandas as pd
import json
from collections import defaultdict
from app.features import extract_features

def train_model(fake_csv="data/Fake.csv", real_csv="data/True.csv", output="model.json"):
    fake_df = pd.read_csv(fake_csv)
    real_df = pd.read_csv(real_csv)
    fake_counts = defaultdict(int)
    real_counts = defaultdict(int)
    fake_docs = len(fake_df)
    real_docs = len(real_df)

    for text in fake_df['text']:
        feats = extract_features(text)
        for k,v in feats.items():
            fake_counts[k] += v

    for text in real_df['text']:
        feats = extract_features(text)
        for k,v in feats.items():
            real_counts[k] += v

    all_features = set(list(fake_counts.keys()) + list(real_counts.keys()))
    feature_probs = {"Fake": {}, "Real": {}}
    alpha = 1

    for f in all_features:
        feature_probs["Fake"][f] = (fake_counts.get(f,0)+alpha) / (sum(fake_counts.values()) + alpha*len(all_features))
        feature_probs["Real"][f] = (real_counts.get(f,0)+alpha) / (sum(real_counts.values()) + alpha*len(all_features))

    P_Fake = fake_docs / (fake_docs + real_docs)
    P_Real = real_docs / (fake_docs + real_docs)

    with open(output, "w") as f:
        json.dump({
            "feature_probs": feature_probs,
            "P_Fake": P_Fake,
            "P_Real": P_Real
        }, f)

if __name__ == "__main__":
    train_model()
