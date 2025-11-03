from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('combined_model.joblib')

# Define valid feature names in the same order as training
valid_features = [
    'FUGITIVE', 'STACK', 'WATER1', 'LANDFILL', 'POTW',
    'MAX ONSITE', 'OFF STE REL1', 'OFF STE REL2'
]

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Toxic Release Danger Level Prediction API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check for missing features
        if not all(feature in data for feature in valid_features):
            return jsonify({
                "error": f"Missing features. Required: {valid_features}"
            }), 400

        # Convert input into a DataFrame
        input_df = pd.DataFrame([data])

        # Predict using the loaded model
        prediction = model.predict(input_df)[0]

        return jsonify({
            "input": data,
            "predicted_danger_level": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
