import os
import joblib
from flask import Flask, render_template, request
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Get the current working directory
project_dir = os.getcwd()

# Load the trained model (adjust the path if necessary)
model_path = os.path.join(project_dir, 'model', 'emissions_model.pkl')
model = joblib.load(model_path)

# Mapping for one-hot encoding of fuel types
fuel_type_map = {
    'Petrol': [1, 0, 0],
    'Diesel': [0, 1, 0],
    'Electric': [0, 0, 1],
    'Hybrid': [0, 0, 0]
}

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        engine_size = float(request.form.get('engine_size'))
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        cylinder = int(request.form.get('cylin'))
        fuel_consumption = float(request.form.get('fuel_consumption'))

        # Validate the fuel type
        if fuel_type not in fuel_type_map:
            return render_template('index.html', result="Invalid fuel type!")

        # Encode the fuel type
        fuel_encoded = fuel_type_map[fuel_type]

        # Combine all input features
        input_features = [engine_size, year, fuel_consumption, cylinder] + fuel_encoded

        # Make prediction using the model
        prediction = model.predict([input_features])[0]

        # Format the result
        result = f"{prediction:.2f} kg/year"

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
