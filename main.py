import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib 
import pprint
import sklearn
import cPickle
from sklearn.datasets import fetch_20newsgroups
import json

app = Flask(__name__)

#fetch datasets
x = fetch_20newsgroups(subset='train')

#vectorize the datasets
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vector_train = vectorizer.fit_transform(x.data)

"""# the model
with open('model/my_dumped_classifier.pkl', 'rb') as fid:
    clf = cPickle.load(fid)"""


@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json

            y = json_['pages']
            
            #test the model
            vector_test = vectorizer.transform(y)
            pred = clf.predict(vector_test)

            #prediction = list(clf.predict(query))
            
            politics=0.0
            sports=0.0
            entertainment=0.0
            tech=0.0
            business=0.0

            pred_cat = []
            pred = list(pred)
            for i in range(len(pred)):
                if pred[i] == 1 or pred[i] == 16 or pred[i] == 17 or pred[i] == 18 or pred[i] == 19 or pred[i] ==20:
                    politics = politics+1
                if pred[i] == 2 or pred[i] == 3 or pred[i] == 6 or pred[i] == 12 or pred[i] == 15:
                    tech = tech+1
                if pred[i] == 7:
                    business = business +1
                if pred[i] == 8:
                    sports = sports +1
                if pred[i] == 9 or pred[i] == 10 or pred[i] == 11:
                    entertainment = entertainment + 1
                    sports= sports+1
                if pred[i] == 4 or pred[i] == 5 or pred[i] == 13:   
                    tech = tech +1
                    business = business +1
            total = [entertainment, business , politics, tech,sports]
            imax = max(total)
            entertainment = (entertainment / imax) * 5
            politics = (politics/imax)* 5
            business = (business /imax) * 5
            sports = (sports/imax)* 5
            tech = (tech / imax) * 5
            mydic=dict()
            mydic["politics"] = politics
            mydic["business"] = business
            mydic["entertainment"] =entertainment
            mydic["sports"] = sports
            mydic["tech"] = tech

            return jsonify(mydic)

        except Exception, e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print 'train first'
        return 'no model here'


@app.route('/train', methods=['GET'])
def train():
    # using random forest as an example
    # can do the training separately and just update the pickles
    """from sklearn.ensemble import RandomForestClassifier as rf

    df = pd.read_csv(training_data)
    df_ = df[include]

    categoricals = []  # going to one-hot encode categorical variables

    for col, col_type in df_.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    global clf
    clf = rf()
    start = time.time()
    clf.fit(x, y)
    print 'Trained in %.1f seconds' % (time.time() - start)
    print 'Model training score: %s' % clf.score(x, y)

    joblib.dump(clf, model_file_name)"""

    #fetch datasets
    x = fetch_20newsgroups(subset='train')


    #vectorize the datasets
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vector_train = vectorizer.fit_transform(x.data)

    #create and train the model
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics
    clf = MultinomialNB(alpha=.01)
    clf.fit(vector_train,x.target)

    #save the model
    with open('model/my_dumped_classifier.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)

    return 'Success'


@app.route('/wipe', methods=['GET'])
def wipe():
    """try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception, e:
        print str(e)"""
    return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception, e:
        port = 8080

    try:
        # the model
        with open('model/my_dumped_classifier.pkl', 'rb') as fid:
            clf = cPickle.load(fid)

    except Exception, e:
        print 'No model here'
        print 'Train first'
        print str(e)
        clf = None

    app.run(host='127.0.0.1', port=port, debug=True)
