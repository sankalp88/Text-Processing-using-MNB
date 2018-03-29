# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 19:41:37 2018

@author: hp
"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
df = pd.read_csv("amazon_cells_labelled.txt",sep="\t", header=None)
df.head()
df_x = df[0]
df_y = df[1]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 0)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy= accuracy_score(test_labels, predictions)
    cm=confusion_matrix(test_labels, predictions)
    print('Confusion Matrix')
    print(cm)
    print('\n')
    print('Classification Report')
    print('\n')
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']
    print(classification_report(test_labels, predictions, target_names=target_names))
    print('Accuracy')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('\n')
    
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

from sklearn.naive_bayes import MultinomialNB
mlb = MultinomialNB()
parameters={"alpha": [0.5,0.7,1], "fit_prior": ['True','False']}
clf = GridSearchCV(mlb, parameters)
clf.fit(X_train_tfidf,y_train)

X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

evaluate(clf, X_new_tfidf, y_test)

print('Best parameters')
print(clf.best_params_)


