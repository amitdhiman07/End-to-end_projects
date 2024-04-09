from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("diabetes_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])

        # Make prediction
        prediction = model.predict([[glucose, bmi, age]])

        # Return prediction to the user
        if prediction[0] == 1:
            result = 'diabetic'
        else:
            result = 'not diabetic'

        return render_template('index.html', prediction_text='The patient is {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
