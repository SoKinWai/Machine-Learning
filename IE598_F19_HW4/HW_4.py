
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame as Housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn import datasets, linear_model
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


Housing = pd.read_excel('D:\Machine Learning part 1\HW_4\housing.xlsx')
Housing.head()

print('Rows of data:', Housing.shape[0])
print('Columns of data:', Housing.shape[1])

Housing.info()
Housing.dropna(inplace=True)
Housing.info()
Housing.describe()

stats.probplot(Housing.MEDV,dist="norm",plot=plt)
plt.show()

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(Housing[cols], height=2.5)
plt.show()

corMat= pd.DataFrame(Housing.corr())
corMat

cm = np.corrcoef(Housing[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.show()

Housing2=pd.read_csv('D:\Machine Learning part 1\HW_4\housing2.csv')
Housing2.head()
print('Rows of data:', Housing2.shape[0])
print('Columns of data:', Housing2.shape[1])

Housing2.info()
Housing2.dropna(inplace=True)
Housing2.info()
Housing2.describe()

#spilt train and test data 
X = Housing.iloc[:, :-1].values
y = Housing['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Shape of X_train: ', X_train.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of X_test: ', X_test.shape)
print('Shape of y_test: ', y_test.shape)

# Linear Regression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)
plt.scatter(y_train_pred,  y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

print('intercept=',regressor.intercept_)
print('coef =', regressor.coef_)
print('training set R2 =', regressor.score(X_train,y_train))
print('trainging set MSE =', mean_squared_error(regressor.predict(X_train),y_train))
print('test set R2=' , regressor.score(X_test, y_test))
print('test set MSE=', mean_squared_error(regressor.predict(X_test),y_test))

# LASSO
alpha=np.arange(0.001,0.05,0.001)
min_test_mse_list=[]
lasso_list=[]
lasso_inter=[]
for i in alpha:
    lasso = Lasso(alpha=i)
    lasso.fit(X_train,y_train)
    min_test_mse_list.append(mean_squared_error(lasso.predict(X_test), y_test))
    lasso_list.append(lasso.coef_)
    lasso_inter.append(lasso.intercept_)
    train_R2 = lasso.score(X_train, y_train)
    test_R2 = lasso.score(X_test, y_test)
    train_mse = mean_squared_error(lasso.predict(X_train), y_train)

index=min_test_mse_list.index(min(min_test_mse_list))
print('The lowest test MSE is ',min(min_test_mse_list))
lasso_coef = lasso_list[index]
print('cofficient=',lasso_coef)
intercept = lasso_inter[index]
print('intercept: ', intercept)
print('Alpha is',alpha[index])
print('testing set R2',train_R2)
print('training set MSE: ', train_mse)
print('testing set R2: ', test_R2)
print('testing set MSE: ',min(min_test_mse_list))

lasso.predict(X_test)-y_test
plt.scatter(y_train_pred,  y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

# Ridge
alpha=np.arange(0.001,1,0.001)
min_test_mse_list=[]
ridge_list=[]
ridge_inter=[]
for i in alpha:
    ridge = Ridge(alpha=i)
    ridge.fit(X_train,y_train)
    min_test_mse_list.append(mean_squared_error(ridge.predict(X_test), y_test))
    ridge_list.append(ridge.coef_)
    ridge_inter.append(lasso.intercept_)
    train_R2 = ridge.score(X_train, y_train)
    test_R2 = ridge.score(X_test, y_test)
    train_mse = mean_squared_error(ridge.predict(X_train), y_train)

index=min_test_mse_list.index(min(min_test_mse_list))
print('The lowest test MSE is ',min(min_test_mse_list))
ridge_coef = ridge_list[index]
print('cofficient=',ridge_coef)
intercept = ridge_inter[index]
print('intercept: ', intercept)
print('Alpha is',alpha[index])
print('testing set R2',train_R2)
print('training set MSE: ', train_mse)
print('testing set R2: ', test_R2)
print('testing set MSE: ',min(min_test_mse_list))

ridge.predict(X_test)-y_test
plt.scatter(y_train_pred,  y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

#To be honest, I do not really understand how these codes work. I had to ask my classmates and googled them online to do my HW 4
print("My name is Jianwei Su")
print("My NetID is: jianwei5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")