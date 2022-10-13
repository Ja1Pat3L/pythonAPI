
from urllib import request
import joblib as jb
from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd
import json
import sys
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics

app = Flask(__name__)
ap = ""
name = ""
# y_train = jb.load('y_train.pkl')
# y_test = jb.load('y_test.pkl')
y_test1 = ""
accuracy = ""
y_train_predict = ""


@app.route("/", methods=["GET", "POST"])
def Fun_knn():
    return render_template("index.html")


@app.route("/sub", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        input_dict = request.form.to_dict()
        
        
        int_features = pd.DataFrame.from_dict([input_dict])
        print(int_features)
        int_features['Occurrence_DayOfMonth'] = int_features['Occurrence_DayOfMonth'].astype(int)
        int_features['Occurrence_DayOfYear'] = int_features['Occurrence_DayOfYear'].astype(int)
        int_features['Occurrence_Hour'] = int_features['Occurrence_Hour'].astype(int)
        int_features['Report_Year'] = int_features['Report_Year'].astype(int)
        int_features['Report_DayOfMonth'] = int_features['Report_DayOfMonth'].astype(int)
        int_features['Report_DayOfYear'] = int_features['Report_DayOfYear'].astype(int)
        int_features['Report_Hour'] = int_features['Report_Hour'].astype(int)
        int_features['Bike_Speed'] = int_features['Bike_Speed'].astype(int)
        int_features['Cost_of_Bike'] = int_features['Cost_of_Bike'].astype(int)
        categorical_cols = [col for col in int_features.columns if int_features[col].dtype == 'object']
        for i in categorical_cols[1:]:
            
            
            with open('Model/'+i+".pkl", 'rb') as f:
                le = pickle.load(f)
                int_features[i] = le.fit_transform(int_features[i].values)
        # print(int_features)
        print(f'****************{int_features.columns}')
        with open('model.pkl', 'rb') as f:
            clf = pickle.load(f)
        output = clf.predict(int_features).tolist()[0]
        with open('x_test.pkl', 'rb') as x:
            x_test = pickle.load(x)
        with open('y_test.pkl', 'rb') as y:
            y_test = pickle.load(y)
        accuracy = round(metrics.accuracy_score(y_test, clf.predict(x_test)),2)*100
        print(f'******************{output}')

        if output == 0:
            return render_template("sub.html", prediction_text="Bike will not be recovered", accuracy=accuracy)
        else:
            return render_template("sub.html", prediction_text="Happy to say, bike will be recovered", accuracy=accuracy)


if __name__ == "__main__":
    app.run(debug=True)
