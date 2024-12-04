from flask import Flask, render_template, request, redirect
import joblib
from data_loader import DataLoader
from data_visualizer import DataVisualizer
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from predictor import Predictor
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('wine_quality_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form_page():
    return render_template('form.html')

@app.route('/upload_csv')
def upload_csv():
    return render_template('upload_csv.html')

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        data = pd.read_csv(file)

        required_features = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]

        if not all(feature in data.columns for feature in required_features):
            return "CSV file is missing some required features."

        data = data[required_features]  # Select only the required columns

        # Make predictions
        predictions = []
        for index, row in data.iterrows():
            input_data = row.values.reshape(1, -1)
            prediction = model.predict(input_data)
            output = 'Good Quality Wine' if prediction[0] == 1 else 'Bad Quality Wine'
            predictions.append(output)

        data.insert(0, 'Serial No.', range(1, len(data) + 1))
        data['Prediction'] = predictions

        def add_classes(row):
            css_class = 'good-quality' if row['Prediction'] == 'Good Quality Wine' else 'bad-quality'
            row_html = ''.join([f'<td class="{css_class}">{val}</td>' for val in row])
            return f'<tr>{row_html}</tr>'

        table_html = '<table class="dataframe">'
        table_html += '<thead><tr>' + ''.join([f'<th>{col}</th>' for col in data.columns]) + '</tr></thead>'
        table_html += '<tbody>' + ''.join(data.apply(add_classes, axis=1)) + '</tbody>'
        table_html += '</table>'

        return render_template('upload_csv.html', table_html=table_html)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    output = 'Good Quality Wine' if prediction[0] == 1 else 'Bad Quality Wine'
    return render_template('form.html', prediction_text= output)

if __name__ == "__main__":
    app.run(debug=True)
