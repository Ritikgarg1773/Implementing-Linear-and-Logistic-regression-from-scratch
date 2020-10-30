import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from scratch import MyPreProcessor
from sklearn import metrics

preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(3)
# logisticRegr = LogisticRegression(penalty='none', tol=0.0001,max_iter=1000)
sdg_reg = SGDClassifier(loss='log',penalty = 'none',fit_intercept = True,max_iter=1000,learning_rate='constant',eta0=0.01,early_stopping=False)
# logisticRegr.fit(X_train,y_train)
sdg = sdg_reg.fit(X,y)
print(sdg.coef_)
print(sdg.intercept_)
print(np.exp(sdg.coef_))
X_train = np.array([75,2]).reshape(-1,1).T
print(X_train.shape)
pred_train_sgd = sdg_reg.predict(X_train)
print(pred_train_sgd)
