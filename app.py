from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

# Define routes
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
        data['Prediction'] = data.apply(lambda row: 'Good Quality Wine' if model.predict(np.array(row).reshape(1, -1))[0] == 1 else 'Bad Quality Wine', axis=1)
        return render_template('upload_csv.html', tables=[data.to_html(classes='data', header="true", index=False)], titles=data.columns.values)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    output = 'Good Quality Wine' if prediction[0] == 1 else 'Bad Quality Wine'
    return render_template('form.html', prediction_text=f'Wine Quality: {output}')

if __name__ == "__main__":
    app.run(debug=True)