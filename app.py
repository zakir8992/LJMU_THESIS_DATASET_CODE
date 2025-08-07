from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("rf_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Define expected order of input features
    features = [
        "pH", "Iron", "Nitrate", "Chloride", "Lead", "Zinc",
        "Turbidity", "Fluoride", "Copper", "Odor",
        "Sulfate", "Conductivity", "Chlorine", "Manganese",
        "Total Dissolved Solids", "Water Temperature", "Air Temperature", "Time of Day"
    ]

    input_data = np.array([data[feature] for feature in features]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][int(prediction)]

    # Simulate dummy feature importances
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-3:][::-1]
    top_features = [{"feature": features[i], "importance": round(importances[i], 4)} for i in top_indices]

    return jsonify({
        "prediction": int(prediction),
        "probability": round(float(proba), 4),
        "top_features": top_features
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
