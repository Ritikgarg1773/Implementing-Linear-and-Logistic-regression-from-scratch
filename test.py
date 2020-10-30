from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D

preprocessor = MyPreProcessor()

print('Linear Regression')

X, y = preprocessor.pre_process(0)
numberOfRows = int(X.shape[0]*0.7)
X_train = X[:numberOfRows]
X_test = X[numberOfRows:]
y_train = y[:numberOfRows]
y_test = y[numberOfRows:]

linear = MyLinearRegression()
# Create your k-fold splits or train-val-test splits as required
k = 5
X_folds = np.array_split(X_train,k)
y_folds = np.array_split(y_train,k)
RMSE_final = []
MAE_final = []
for i in range(k):
	X_data_train = X_folds.copy() # here we are working on the copy
	y_data_train = y_folds.copy()
	X_data_test = X_folds[i]
	y_data_test = y_folds[i]
	del X_data_train[i] #delete the ith fold from the X_train_data
	del y_data_train[i] #delete the ith fold from the y_train_data
	X_data_train = np.concatenate(X_data_train) # here we don't want our data to get sorted while concatinating
	y_data_train = np.concatenate(y_data_train) 
	RMSE_errors,MAE_errors, Rmse_coef, Rmse_intercept, Mae_coef, Mae_intercept = linear.fit(X_data_train,y_data_train)   
	# print(Rmse_coef)
	y_predicted = linear.predict(X_data_test)  # Now we have the predicited and the actual values of the test data
	RMSE_validation,MAE_Validation = linear.validationLoss(X_data_test,y_data_test,Rmse_coef,Rmse_intercept,Mae_coef,Mae_intercept)

	plt.plot(RMSE_errors,color='red',linewidth=2,label = 'RMSE Training Loss')  #got the training loss v/s iteartions for RMSE
	plt.plot(RMSE_validation,color='orange',linewidth=2,label = 'RMSE Validation Loss')  #got the training loss v/s iterations for MAE
	plt.xlabel("Number of Iterations")
	plt.ylabel("Loss")
	plt.legend()
	# print('True Values:', ytest)
	plt.show()          #plotted the graph
	plt.clf()
	print('RMSE Error on Test Data: ' + str(RMSE_validation[-1]))
	# print(linear.coefficient,linear.intercept)  #Here are the intercepts and coefficient
	
	plt.plot(MAE_errors,color='red',linewidth=2,label = 'MAE Training Loss')  #got the training loss v/s iteartions for RMSE
	plt.plot(MAE_Validation,color='blue',linewidth=2,label = 'MAE Validation Loss')  #got the training loss v/s iterations for MAE
	plt.xlabel("Number of Iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()
	plt.clf()
	print('MAE Error on Test Data: ' + str(MAE_Validation[-1]))
	# print(y_predicted)
	# RMSE_final.append(linear.RMSE_errors(X_data_test))
	# MAE_final.append(linear.MAE_errors(X_data_test))
	# print("RMSE error on Test Data : " + str(RMSE_final[-1]))
	# print("MAE error on Test Data : " + str(MAE_final[-1]))

	# plt.plot(y_data_test)
	# plt.plot(y_predicted)
	# plt.show()
# y_pred = linear.predict(X_test)
# RMSE_test_errors, MAE_test_errors = linear.validationLoss(y_test,y_pred)
# plt.plot(RMSE_test_errors,color='red',linewidth=2,label = 'RMSE Loss')
# plt.plot(MAE_test_errors,color='blue',linewidth=2,label = 'MAE Loss')
# plt.xlabel("Iterations")
# plt.ylabel("Validation loss")
# plt.legend()
# # print('True Values:', ytest)
# plt.show()


# feautres = 2
# X_train = X[:,:feautres]
# X_train = 

# Xtrain = np.empty((0,0))
# ytrain = np.empty((0))
# Xtest = np.empty((0,0))
# ytest = np.empty((0))



# linear.fit(Xtrain, ytrain)

# ypred = linear.predict(Xtest)
# ypred = linear.predict(X)
# print('Predicted Values:', ypred)


print('Logistic Regression')

X, y = preprocessor.pre_process(2)
numberOfRows = int(X.shape[0] * 0.1)
X_train = X[:numberOfRows*7]
print(len(X_train))
y_train = y[:numberOfRows*7]
X_eval = X[numberOfRows*7:(numberOfRows)*(8)]
print(len(X_eval))
y_eval = y[numberOfRows*7 : numberOfRows*8]
X_test = X[numberOfRows*8:]
y_test = y[numberOfRows*8:]
print(len(X_test))
# Create your k-fold splits or train-val-test splits as required

# Xtrain = np.empty((0,0))
# ytrain = np.empty((0))
# Xtest = np.empty((0,0))
# ytest = np.empty((0))

logistic = MyLogisticRegression()
Batch_errors, Stochastic_errors,Batch_coef, Batch_intercept, Stochastic_coef, Stochastic_intercept = logistic.fit(X_train, y_train)
Batch_test, Stochastic_test = logistic.validationLoss(X_test,y_test,Batch_coef,Batch_intercept,Stochastic_coef,Stochastic_intercept)

# ypred = logistic.predict(Xtest)
# ypred = logistic.predict(X)
print("Accuracy on Train data using Batch Gradient Descent: " + str(logistic.accuracy(X_train, y_train,Batch_coef[-1],Batch_intercept[-1])))

print("Accuracy on Test data using Batch Gradient Descent: " + str(logistic.accuracy(X_test, y_test,Batch_coef[-1],Batch_intercept[-1])))
# print('Predicted Values:', ypred)
# print('True Values:', ytest)
# print('coefficient intercept')
# print(logistic.coefficient,logistic.intercept)
plt.plot(Batch_errors,color='red',linewidth=1,label = 'Test errors')
plt.plot(Batch_test, color = 'orange',linewidth = 1, label = 'Validation loss')
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.legend()
plt.show()
plt.clf()
print("Accuracy on Train data using Stochastic Gradient Descent: " + str(logistic.accuracy(X_train, y_train,Stochastic_coef[-1],Stochastic_intercept[-1])))

print("Accuracy on Test data using Stochastic Gradient Descent: " + str(logistic.accuracy(X_test, y_test,Stochastic_coef[-1],Stochastic_intercept[-1])))
plt.plot(Stochastic_errors,color='red',linewidth=1,label = 'Test errors')
plt.plot(Stochastic_test, color = 'orange',linewidth = 1, label = 'Validation loss')
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.legend()
plt.show()
plt.clf()
