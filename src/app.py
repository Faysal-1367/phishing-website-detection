from flask import Flask, render_template, request
import joblib
import os

try:
    from src.feature_extraction import extract_features
except:
    from feature_extraction import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "templates")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print("Model Load Error:", e)
    model = None

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    risk_level = None
    submitted_url = None
    recommendations = []

    confidence = 0
    phishing_prob = 0
    legitimate_prob = 0
    feature_dict = None
    result_color = None

    if request.method == "POST":
        if model is None:
            return render_template(
                "index.html",
                result="Model not loaded. Please check model.pkl"
            )
        url = request.form.get("url", "").strip()

        if not url:
            return render_template(
                "index.html",
                result="Please enter a valid URL"
            )

        submitted_url = url
        print("Submitted URL =", submitted_url)
        feature_dict = extract_features(url)

        features = list(feature_dict.values())

        prediction = model.predict([features])[0]

        probabilities = model.predict_proba([features])[0]

        legitimate_prob = round(probabilities[0] * 100, 2)
        phishing_prob = round(probabilities[1] * 100, 2)

        if prediction == 1:
            result = "⚠️ Phishing Website"
            result_color = "red"
            confidence = phishing_prob

            if phishing_prob >= 80:
                risk_level = "HIGH"
            elif phishing_prob >= 50:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            recommendations = [
                "Do not enter passwords",
                "Avoid online payments",
                "Verify the domain name",
                "Check website certificate"
            ]
        else:
            result = "✅ Legitimate Website"
            result_color = "green"
            confidence = legitimate_prob
            risk_level = "LOW"
            recommendations = [
                "HTTPS Enabled",
                "No suspicious symbols found",
                "Safe URL structure",
                "Website appears legitimate"
            ]

    return render_template(
        "index.html",
        result=result,
        result_color=result_color,
        confidence=confidence,
        phishing_prob=phishing_prob,
        legitimate_prob=legitimate_prob,
        features=feature_dict,
        submitted_url=submitted_url,
        risk_level=risk_level,
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)