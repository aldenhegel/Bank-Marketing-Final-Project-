from flask import Flask, request, jsonify, render_template, flash
import requests
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        body = request.json
        print(body)
        nr_employed = float(body['nr.employed'])
        pdays = float(body['pdays']) 
        cons_conf_idx = float(body['cons.conf.idx']) 
        euribor3m = float(body['euribor3m'])
        cons_price_idx = float(body['cons.price.idx'])
        emp_var_rate = float(body['emp.var.rate'])
        previous = float(body['previous'])
        age = float(body['age'])
        campaign = float(body['campaign'])
        education = educationDict[str(body['education'])]

        df = pd.DataFrame(data=[[age,
                                education,
                                campaign,
                                pdays,
                                previous,
                                emp_var_rate,
                                cons_price_idx,
                                cons_conf_idx,
                                euribor3m,
                                nr_employed]], 
                            columns=['age', 'education', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                                        'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
                            )

        proba = round(float(xgb.predict_proba(df)[:, 1]) * 100, 2)

        df['y'] = xgb.predict(df)[0]

        records = pd.read_csv('records.csv').drop(columns='Unnamed: 0')
        records = records.append(df)
        records.to_csv('records.csv')

        return jsonify({
            'nr_employed': nr_employed,
            'pdays': pdays,
            'cons_conf_idx': cons_conf_idx, 
            'euribor3m': euribor3m,
            'cons_price_idx': cons_price_idx,
            'emp_var_rate': emp_var_rate,
            'previous': previous,
            'age': age,
            'campaign': campaign,
            'education': body['education'],
            'proba': proba
        })

@app.route('/input', methods=['GET', 'POST'])
def input():
    if request.method == 'POST':
        body = request.json
        print(body)
        nr_employed = float(body['nr.employed'])
        pdays = float(body['pdays']) 
        cons_conf_idx = float(body['cons.conf.idx']) 
        euribor3m = float(body['euribor3m'])
        cons_price_idx = float(body['cons.price.idx'])
        emp_var_rate = float(body['emp.var.rate'])
        previous = float(body['previous'])
        age = float(body['age'])
        campaign = float(body['campaign'])
        education = educationDict[str(body['education'])]
        y = float(body['y'])

        df = pd.DataFrame(data=[[age,
                                education,
                                campaign,
                                pdays,
                                previous,
                                emp_var_rate,
                                cons_price_idx,
                                cons_conf_idx,
                                euribor3m,
                                nr_employed,
                                y]], 
                            columns=['age', 'education', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                                        'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']
                            )

        new = pd.read_csv('new.csv').drop(columns='Unnamed: 0')
        new = new.append(df)
        new.to_csv('new.csv')

        return jsonify({
            'nr_employed': nr_employed,
            'pdays': pdays,
            'cons_conf_idx': cons_conf_idx, 
            'euribor3m': euribor3m,
            'cons_price_idx': cons_price_idx,
            'emp_var_rate': emp_var_rate,
            'previous': previous,
            'age': age,
            'campaign': campaign,
            'education': body['education'],
            'y': y
        })


if __name__ == '__main__':

    xgb = joblib.load('xgb')

    educationDict = {'basic.4y': 0,
                    'high.school': 3,
                    'basic.6y': 1,
                    'basic.9y': 2,
                    'professional.course': 5,
                    'unknown': 7,
                    'university.degree': 6,
                    'illiterate': 4}

    app.run(debug=True, port=5002)