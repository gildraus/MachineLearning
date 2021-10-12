import pandas as pd
from sklearn import datasets

diabetes = datasets.load_diabetes()

X = diabetes.data
Y = diabetes.target
#isto moze i ovako
X, Y = datasets.load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#definisanje modela
model = linear_model.LinearRegression()
#kreiranje modela
model.fit(X_train, Y_train)

#kreiramo predikcije
Y_pred = model.predict(X_test)

#ispisujemo neke metrike
print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)
print('Mean squared error(MSE): %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coeficient of determination (R^2): %.2f' % r2_score(Y_test, Y_pred))

import seaborn as sns

#print(Y_test)
#print(Y_pred)

#sns.scatterplot(Y_test, Y_pred, marker = "+")

boston = pd.read_csv("boston.csv")

print(boston)

Y = boston.medv
print(Y)

X = boston.drop(['medv'], axis = 1)
print(X)

#ostalo je sve isto


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#definisanje modela
model = linear_model.LinearRegression()
#kreiranje modela
model.fit(X_train, Y_train)

#kreiramo predikcije
Y_pred = model.predict(X_test)

#ispisujemo neke metrike
print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)
print('Mean squared error(MSE): %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coeficient of determination (R^2): %.2f' % r2_score(Y_test, Y_pred))



sns.scatterplot(Y_test, Y_pred)

print("pauza")