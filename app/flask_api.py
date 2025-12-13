from flask import Flask, request, jsonify, render_template
from model import predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("frontend.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400

    result = predict(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
