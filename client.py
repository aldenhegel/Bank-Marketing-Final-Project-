from flask import Flask, request, jsonify, render_template, flash
import requests

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')

@app.route('/predictform', methods=['POST', 'GET'])
def predictform():
    if request.method == 'POST':
        data = request.form 

        url = 'http://127.0.0.1:5002/predict'

        try:
            prediction = requests.post(url, json=data).json()
            return render_template('predictresult.html', prediction=prediction)
        except:
            return render_template('error.html')
    else:
        return render_template('predictform.html')

@app.route('/inputform', methods=['POST', 'GET'])
def inputform():
    if request.method == 'POST':
        data = request.form 

        url = 'http://127.0.0.1:5002/input'

        data_input = requests.post(url, json=data).json()
        return render_template('inputresult.html', prediction=data_input)
    else:
        return render_template('inputform.html')
        
if __name__ == '__main__':
    app.run(debug=True, port=5001)