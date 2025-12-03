from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load model and pipeline
model = pickle.load(open('model.pkl', 'rb'))
pipeline = pickle.load(open('pipeline.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from form
    input_data = [request.form.get(col) for col in ['income', 'loan_amount', 'term_length', 'occup', 'marital', 'schufa', 'num_applic', 'install_to_inc']]
    
    # Convert to appropriate data types
    df = pd.DataFrame([input_data], columns=['income', 'loan_amount', 'term_length', 'occup', 'marital', 'schufa', 'num_applic', 'install_to_inc'])
    df = df.astype({
        'income': float,
        'loan_amount': float,
        'term_length': float,
        'occup': str,
        'marital': str,
        'schufa': float,
        'num_applic': float,
        'install_to_inc': float
    })

    # Apply pipeline and predict
    X = pipeline.transform(df)
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    result = "Defaulter" if prediction == 1 else "Non-defaulter"
    return render_template('index.html', prediction_text=f"{result} (Risk Probability: {prob:.2f})")

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 8000, debug = True)
