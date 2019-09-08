# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 07:17:13 2019

@author: Jianwei Su
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree

treasury = pd.read_csv("D:\Machine Learning part 1\HW_2\Treasury Squeeze test - DS1.csv",header = None)

treasury = np.array(treasury)
treasury = np.delete(treasury, 0, axis = 0)

treasury = treasury[:,2:12]

treasury_data=treasury[:,0:8]
treasury_target=treasury[:,9]

X_train, X_test, y_train, y_test = train_test_split(treasury_data, treasury_target, test_size=0.3, random_state=3, stratify=treasury_target)

#try K=1 through K=25 and record testing accuracy
k_range = range(1, 26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
for k in k_range:
    print ("k =",k,' ',"Accuracy =",scores[k-1]*100,"%") 

plt.title('KNN with different k')
plt.plot(k_range,scores)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

treasury_data=treasury[:,0:9]
treasury_target=treasury[:,9]

print(treasury_data[0:3,:])
print(treasury_target[0:3])

X_train, X_test, y_train, y_test = train_test_split(treasury_data, treasury_target, test_size=0.3, random_state=3, stratify=treasury_target)
TreasuaryTreeModel = tree.DecisionTreeClassifier(criterion= 'gini',max_depth=3,random_state=1)
TreasuaryTreeModel.fit(X_train, y_train)

y_predict = TreasuaryTreeModel.predict(X_test)
print(accuracy_score(y_test, y_predict))

tree.plot_tree(TreasuaryTreeModel)

print("My name is Jianwei Su")
print("My NetID is: jianwei5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
#To be honest, I don't know how to do this homework assignment. I googled that and asked my classmates, I still don't understand now.