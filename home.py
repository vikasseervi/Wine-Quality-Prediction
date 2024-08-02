from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def page1():
    return render_template('form.html')

@app.route('/upload csv')
def page2():
    return render_template('upload_csv.html')

if __name__ == '__main__':
    app.run(debug=True)
