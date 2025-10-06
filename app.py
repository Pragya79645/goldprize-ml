from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load model
model = pickle.load(open("regressor_model.pkl", "rb"))

app = Flask(__name__)
CORS(app)  # allow frontend requests

@app.route('/')
def home():
    return "Gold Price Prediction API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
