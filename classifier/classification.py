import numpy as np
import pandas as pd

df = pd.read_csv('./sf_fire_deparment_submit.csv')
df['date'] = pd.to_datetime(df['Received DtTm'], format='%m/%d/%Y %I:%M:%S %p')
df['hour'] = df['date'].dt.hour
df['dayOfWeek'] = df['date'].dt.dayofweek
df = df[['Zipcode of Incident', 'dayOfWeek','Supervisor District','hour', 'Call Type']]
hotkeydf = pd.get_dummies(data=df, columns=['Zipcode of Incident', 'dayOfWeek','Supervisor District'])
hotkeydf['hour'] = df['hour'] # append non hotkey attribute

from sklearn.model_selection import train_test_split
train, test = train_test_split(hotkeydf, test_size=0.2)
# x is feature, y is label
train_x = train.copy().drop('Call Type', 1)
train_y = train[['Call Type']]
test_x = test.copy().drop('Call Type', 1)
test_y = test[['Call Type']]

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
logisticRegression = LogisticRegression(random_state=1)
randomForest = RandomForestClassifier(random_state=1)
neuralNetwork = MLPClassifier(alpha = 1)
voting = VotingClassifier(estimators=[('lr', logisticRegression), ('rf', randomForest), ('nn', neuralNetwork)],voting='soft')
classifierList = [sgd, logisticRegression, randomForest, neuralNetwork, voting]
classifierNameList = ["SGD", "logisticRegression", "randomForest", "neuralNetwork", "voting"]

le = preprocessing.LabelEncoder()
le.fit_transform(train_y['Call Type'])

def trainModelAndPrintResult(classifer, classifier_name):
    model = classifer.fit(train_x, le.fit_transform(train_y))
    train_score = model.score(train_x, le.transform(train_y))
    test_score = model.score(test_x, le.transform(test_y))
    print("Classification Score for: ", classifier_name)
    print("Trainning score: ", train_score)
    print("Test score: ", test_score)

for i in range(len(classifierList)):
	trainModelAndPrintResult(classifierList[i], classifierNameList[i])




