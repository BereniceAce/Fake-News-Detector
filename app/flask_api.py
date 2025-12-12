from flask import Flask, request, jsonify, render_template
from model import predict, update_model, compute_model
#from features import extract_features

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def home():
    return render_template("frontend.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = data.get("text", "").strip()
    url = data.get("url", "").strip()

    if not text:
        return jsonify({"error": "Text is required."}), 400

    result = predict(text, url)

    # Prepare score breakdown for frontend
    feats = result["features"]
    breakdown = {
        "real_keyword_pts": feats.get("real_kw", 0),
        "fake_keyword_pts": feats.get("fake_kw", 0),
        "punct_good": 0 if feats.get("punct_bad", 0) else 1,
        "punct_bad": feats.get("punct_bad", 0),
        "spell_good": 0 if feats.get("spell_bad", 0) else 1,
        "spell_bad": feats.get("spell_bad", 0),
        "url_good": feats.get("url_good", 0),
        "url_bad": feats.get("url_bad", 0),
        "real_score_total": feats.get("real_kw", 0),
        "fake_score_total": feats.get("fake_kw", 0)
    }

    response = {
        "result": result["verdict"],
        "final_score": round(result["probability"] * 100, 2),
        "features": feats,
        "score_breakdown": breakdown,
        "url_status": "Valid" if feats.get("url_good", 0) else "Suspicious"
    }
    return jsonify(response)

@app.route("/feedback", methods=["POST"])
def feedback_route():
    data = request.get_json()
    text = data.get("text", "").strip()
    url = data.get("url", "").strip()
    label = data.get("label", "").strip().lower()

    if not text or label not in ["real", "fake"]:
        return jsonify({"error": "Text and valid label ('real' or 'fake') are required."}), 400

    update_model(text, url, label)

    return jsonify({"status": "success", "message": "Model updated with feedback."})

if __name__ == "__main__":
    # Ensure model is initialized
    compute_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
