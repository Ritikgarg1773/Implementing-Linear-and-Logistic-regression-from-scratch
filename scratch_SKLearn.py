import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from scratch import MyPreProcessor
from sklearn import metrics

preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(3)
numberOfRows = int(X.shape[0] * 0.1)
X_train = X[:numberOfRows*7]
# print(len(X_train))
y_train = y[:numberOfRows*7]
X_eval = X[numberOfRows*7:(numberOfRows)*(8)]
# print(len(X_eval))
y_eval = y[numberOfRows*7 : numberOfRows*8]
X_test = X[numberOfRows*8:]
y_test = y[numberOfRows*8:]
# print(len(X_test))

# x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.25, random_state=0)
# logisticRegr = LogisticRegression(penalty='none', tol=0.0001,max_iter=1000)
sdg_reg = SGDClassifier(loss='log',penalty = 'none',fit_intercept = True,max_iter=100,learning_rate='constant',eta0=0.01,early_stopping=False)
# logisticRegr.fit(X_train,y_train)
sdg_reg.fit(X_train,y_train)
# pred_train_logistic = logisticRegr.predict(X_train)
# accuracy_train = metrics.accuracy_score(y_train, pred_train_logistic)
# print("Accuracy on train using LogisticRegression: " + str(accuracy_train)) 
# print(logisticRegr)
pred_train_sgd = sdg_reg.predict(X_train)
acc_train_sgd = metrics.accuracy_score(y_train,pred_train_sgd)
print("Accuracy on train using SDG: " + str(acc_train_sgd*100)+"%")
# predictions = logisticRegr.predict(X_test)
pred_sdg = sdg_reg.predict(X_test)
# score = logisticRegr.score(X_test, y_test)
# print(score)
# accuracy1 = metrics.accuracy_score(y_test, predictions)
# accuracy_percentage = 100 * accuracy
# print("Accuracy on test data using LogisticRegression"+ str(accuracy1))
accuracy2 = metrics.accuracy_score(y_test, pred_sdg)
# accuracy_percentage = 100 * accuracy
print("Accuracy on test data using SDG: "+str(accuracy2*100)+"%")
