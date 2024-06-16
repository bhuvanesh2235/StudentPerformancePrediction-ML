from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import os
import warnings as w
w.filterwarnings('ignore')

app = Flask(__name__)

data = pd.read_csv("AI-Data.csv")
data = data.drop(columns=["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth", "SectionID", "Topic", "Semester", "Relation", "ParentCollegeSatisfaction", "ParentAnsweringSurvey", "AnnouncementsView"])
data = u.shuffle(data)

# Encode categorical data
for column in data.columns:
    if data[column].dtype == type(object):
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# Split data
ind = int(len(data) * 0.70)
feats = data.values[:, 0:4]
lbls = data.values[:, 4]
feats_train = feats[:ind]
feats_test = feats[ind:]
lbls_train = lbls[:ind]
lbls_test = lbls[ind:]

# Train models
modelD = tr.DecisionTreeClassifier().fit(feats_train, lbls_train)
modelR = es.RandomForestClassifier().fit(feats_train, lbls_train)
modelP = lm.Perceptron().fit(feats_train, lbls_train)
modelL = lm.LogisticRegression().fit(feats_train, lbls_train)
modelN = nn.MLPClassifier(activation="logistic").fit(feats_train, lbls_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'raised_hands': int(request.form['raised_hands']),
        'visited_resources': int(request.form['visited_resources']),
        'discussions': int(request.form['discussions']),
        'absences': int(request.form['absences'])
    }
    features = np.array([
        input_data['raised_hands'],
        input_data['visited_resources'],
        input_data['discussions'],
        input_data['absences']
    ]).reshape(1, -1)

    predD = modelD.predict(features)[0]
    predR = modelR.predict(features)[0]
    predP = modelP.predict(features)[0]
    predL = modelL.predict(features)[0]
    predN = modelN.predict(features)[0]

    result = {
        "Decision Tree": "H" if predD == 0 else "M" if predD == 1 else "L",
        "Random Forest": "H" if predR == 0 else "M" if predR == 1 else "L",
        "Perceptron": "H" if predP == 0 else "M" if predP == 1 else "L",
        "Logistic Regression": "H" if predL == 0 else "M" if predL == 1 else "L",
        "Neural Network": "H" if predN == 0 else "M" if predN == 1 else "L"
    }

    return render_template('index.html', result=result)

@app.route('/graph', methods=['POST'])
def graph():
    graph_choice = int(request.form['graph_choice'])
    graph_path = None

    if graph_choice == 1:
        sb.countplot(x='Class', data=data, order=['L', 'M', 'H'])
        plt.title('Marks Class Count Graph')
        graph_path = 'graph1.png'
    elif graph_choice == 2:
        sb.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title('Marks Class Semester-wise Graph')
        graph_path = 'graph2.png'
    elif graph_choice == 3:
        sb.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H'])
        plt.title('Marks Class Gender-wise Graph')
        graph_path = 'graph3.png'
    elif graph_choice == 4:
        sb.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title('Marks Class Nationality-wise Graph')
        graph_path = 'graph4.png'
    elif graph_choice == 5:
        sb.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order=['L', 'M', 'H'])
        plt.title('Marks Class Grade-wise Graph')
        graph_path = 'graph5.png'
    elif graph_choice == 6:
        sb.countplot(x='SectionID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title('Marks Class Section-wise Graph')
        graph_path = 'graph6.png'
    elif graph_choice == 7:
        sb.countplot(x='Topic', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title('Marks Class Topic-wise Graph')
        graph_path = 'graph7.png'
    elif graph_choice == 8:
        sb.countplot(x='StageID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title('Marks Class Stage-wise Graph')
        graph_path = 'graph8.png'
    elif graph_choice == 9:
        sb.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title('Marks Class Absent Days-wise Graph')
        graph_path = 'graph9.png'

    plt.savefig(os.path.join('static', graph_path))
    plt.close()

    return render_template('graph.html', graph_path=graph_path)

if __name__ == '__main__':
    app.run(debug=True)
