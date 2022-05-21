import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
import pickle
from sklearn.preprocessing import LabelEncoder
import pickle

import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "a0FHHGae4F6CFpI_7sydf0vuNpg0PuvgeGkNbuVKb_5v"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

le = LabelEncoder()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('resaleintro.html')


@app.route('/predict')
def predict():
    return render_template('resalepredict.html')


@app.route('/y_predict', methods=['GET','POST'])
def y_predict():
    regyear = int(request.form['regyear'])
    powerps = float(request.form['powerps'])
    kms = float(request.form['kms'])
    regmonth = int(request.form.get('regmonth'))
    gearbox = request.form['gearbox']
    damage = request.form['dam']
    model = request.form.get('modeltype')
    brand = request.form.get('brand')
    fuelType = request.form.get('fuel')
    vehicletype = request.form.get('vehicletype')


    new_row = {'yearOfRegistration':regyear, 'powerPS':powerps, 'kilometer':kms,
       'monthOfRegistration':regmonth, 'gearbox':gearbox, 'notRepairedDamage':damage,
       'model':model, 'brand':brand, 'fuelType':fuelType,
       'vehicleType':vehicletype}
    print('newRow',new_row)

    """total = [[int(vehicletype), int(regyear), int(gearbox), int(powerps), int(model), int(kms), int(regmonth), int(fuelType),
               int(brand), int(damage)]]
    print(total)"""
    new_df = pd.DataFrame(columns=['vehicleType', 'yearOfRegistration', 'gearbox',
                                   'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType',
                                   'brand', 'notRepairedDamage'])

    new_df = new_df.append(new_row, ignore_index=True)

    lb = LabelEncoder()
    new_df['vehicleType'] = lb.fit_transform([new_df['vehicleType']])
    new_df['gearbox'] = lb.fit_transform([new_df['gearbox']])
    new_df['model'] = lb.fit_transform([new_df['model']])
    new_df['fuelType'] = lb.fit_transform([new_df['fuelType']])
    new_df['brand'] = lb.fit_transform([new_df['brand']])
    new_df['notRepairedDamage'] = lb.fit_transform([new_df['notRepairedDamage']])
    print(new_df.values.tolist())


    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"field": [["vehicleType", "yearOfRegistration", "gearbox",
                                   "powerPS", "model", "kilometer", "monthOfRegistration", "fuelType",
                                   "brand", "notRepairedDamage"]],
                                       "values": new_df.values.tolist()}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/6f07c895-833a-41d2-be6b-e59677fd3d78/predictions?version=2022-03-28', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
    pred = response_scoring.json()
    prediction = pred['predictions'][0]['values'][0][0]

    print(prediction)
    return render_template('resalepredict.html', ypred='The resale value predicted is {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=False)



