# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:08:29 2019

@author: sjw901112
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('D:\Machine Learning part 1\HW_6\ccdefault.csv')
X=df.drop(['ID','DEFAULT'],axis=1).values
y=df['DEFAULT'].values
print(X.shape,y.shape)
print(X[0], y[0])

#Decision Tree
scores_train=[]
scores_test=[]
tree = DecisionTreeClassifier(criterion='gini',max_depth=20)
start = time.clock()
for a in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=a)
    tree.fit(X_train,y_train)
    y_train_pred1 = tree.predict(X_train)
    y_pred1 = tree.predict(X_test)
    scores_train.append(metrics.accuracy_score(y_train, y_train_pred1))
    scores_test.append(metrics.accuracy_score(y_test, y_pred1))
end = time.clock()
print('Run time: ', end - start, 's')
print(scores_train)
print(scores_test)
print(np.mean(scores_train))
print(np.std(scores_train))
print(np.mean(scores_test))
print(np.std(scores_test))
plt.title('Holdout Score')
plt.plot(range(1,11),scores_train)
plt.plot(range(1,11),scores_test)
plt.legend(['train', 'test'])
plt.xlabel('random state')
plt.ylabel('Accuracy')
plt.show()

#K_fold CV
skf = StratifiedKFold(n_splits=10)
start = time.clock()
scores = cross_validate(tree,X,y,cv=skf,scoring='accuracy',return_train_score=True)
end = time.clock()
print('Run time: ', end - start, 's')
print(scores.keys())
print(scores['train_score'])
print(np.mean(scores['train_score']))
print(np.std(scores['train_score']))
print(scores['test_score'])
print(np.mean(scores['test_score']))
print(np.std(scores['test_score']))
plt.title('K_fold CV Score')
plt.plot(range(1,11),scores['train_score'])
plt.plot(range(1,11),scores['test_score'])
plt.legend(['train', 'test'])
plt.xlabel('random state')
plt.ylabel('Accuracy')
plt.show()

#I asked my classmates to finish this assignment
print("My name is Jianwei Su")
print("My NetID is: jianwei5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")