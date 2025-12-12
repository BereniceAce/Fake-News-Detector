import json
from app.features import extract_features

def load_model(model_file="model.json"):
    with open(model_file) as f:
        return json.load(f)

def predict(text, url="", model=None):
    if model is None: model = load_model()
    feats = extract_features(text, url)
    feature_probs = model["feature_probs"]
    P_Fake = model["P_Fake"]
    P_Real = model["P_Real"]

    P_feat_given_fake = 1.0
    P_feat_given_real = 1.0

    for f,v in feats.items():
        if v == 0: continue
        P_feat_given_fake *= feature_probs["Fake"][f] ** v
        P_feat_given_real *= feature_probs["Real"][f] ** v

    prob_fake = (P_feat_given_fake * P_Fake) / ((P_feat_given_fake*P_Fake) + (P_feat_given_real*P_Real))
    verdict = "fake" if prob_fake > 0.5 else "real"
    return {"result": verdict, "prob_fake": prob_fake, "features": feats}

if __name__ == "__main__":
    model = load_model()
    text = "Breaking news: miracle cure for flu exposed!"
    url = "http://example.com/news"
    print(predict(text, url, model))
