from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load pre-trained model and pipeline
model = joblib.load("rf_model.joblib")
pipeline = joblib.load("full_pipeline.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        df_input = pd.DataFrame([data])
        processed = pipeline.transform(df_input)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]
        return jsonify({"prediction": int(prediction), "probability": float(probability)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
