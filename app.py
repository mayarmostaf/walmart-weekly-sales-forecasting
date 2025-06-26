from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

# Load the saved Random Forest model
model = joblib.load("random_forest_model.pkl")

# Sequence length the model expects
N_STEPS = 7

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Random Forest API is running. Use POST /predict with a 7-step sequence.'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sequence = data.get('sequence', None)
        
        if sequence is None:
            return jsonify({'error': 'Missing "sequence" in request.'}), 400
        
        if len(sequence) != 7:
            return jsonify({'error': f'Sequence must be of length 7'}), 400

        # Use feature names
        input_df = pd.DataFrame([sequence])
        prediction = model.predict(input_df)
        return jsonify({'prediction': prediction.flatten().tolist()})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    
if __name__ == '__main__':
    app.run(debug=True)
