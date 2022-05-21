import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
import pickle
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)
rf = pickle.load(open('model.pkl', 'rb'))
le = LabelEncoder()


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
    print(new_row)
    new_df = pd.DataFrame(columns=['vehicleType', 'yearOfRegistration', 'gearbox',
                                   'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType',
                                   'brand', 'notRepairedDamage'])

    new_df = new_df.append(new_row, ignore_index=True)

    def encoder(data, variable):
        lb = LabelEncoder()
        data[variable] = lb.fit_transform(data[variable])

    encoder(new_df, 'vehicleType')
    encoder(new_df, 'gearbox')
    encoder(new_df, 'model')
    encoder(new_df, 'fuelType')
    encoder(new_df, 'brand')
    encoder(new_df, 'notRepairedDamage')

    print(rf.predict(new_df))

    y_prediction = rf.predict(new_df)
    print(y_prediction)
    return render_template('resalepredict.html', ypred='The resale value predicted is {}'.format(y_prediction))


if __name__ == '__main__':
    app.run(debug=False)



"""   new_df = pd.DataFrame(columns =['vehicleType', 'yearOfRegistration', 'gearbox',
                                'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType',
                                'brand', 'notRepairedDamage'] )
    new_df = new_df.append(new_row,ignore_index = True)
    labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
    mapper = {}
    for i in labels:
        mapper[i] = LabelEncoder()
        mapper[i].classes_ = np.load(str('classes'+i+'.npy'))
        tr = mapper[i].fit_transform(new_df[i])
        new_df.loc[:, i + '_labels'] = pd.Series(tr, index=new_df.index)
    labeled = new_df[ ['yearOfRegistration'
                        ,'powerPS'
                        ,'kilometer'
                        ,'monthOfRegistration'
                        ] 
                    + [x+'_labels' for x in labels]]
    X = labeled.values
    print(X)"""