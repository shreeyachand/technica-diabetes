from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
import pandas as pd

app_version = '1.1.0'

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('main.html')

@app.route('/data', methods=["GET", "POST"])
def results():
    if request.method == "POST":
        indicators = ['sex', 'age', 'smoking', 'height', 'weight', 'heart_condition', 'bp', 'cholesterol']
        responses = {x: request.form.get(x) for x in indicators}
        responses['bmi'] = calc_bmi(responses['weight'],responses['height'])
        responses.pop('height')
        responses.pop('weight')
        nn = pickle.load(open('./nnmodel.pkl', 'rb'))
        data = [1 if x=="1" else x for x in responses.values()]
        data = [0 if x==None or x=="0" else x for x in responses.values()]
        outp = np.array([pd.read_csv('y.csv')['Diabetes_012'].iloc[x] for x in nn.kneighbors(np.array(data).reshape(1,-1))[1]])
        return render_template('data_test.html', smok_resp=outp)
    else:
        return redirect('/')


def calc_bmi(lbs, height):
    kg = int(lbs)*0.45359237
    m = int(height) * 0.0254
    return kg / (m*m)

if __name__ == '__main__':
    app.run(debug = True)