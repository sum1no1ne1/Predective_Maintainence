from flask import Flask, request, jsonify
# from flask_ngrok import run_with_ngrok
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
# run_with_ngrok(app)  # Make Flask accessible publicly

# Load trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Flask API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON input
        data = request.get_json()

        # Convert JSON to DataFrame
        df = pd.DataFrame(data["data"], columns=data["columns"])

        # Make predictions
        predictions = model.predict(df)

        return jsonify({"prediction": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
app.run()