import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.datasets import make_regression
# from sklearn.preprocessing import Normalizer
from scipy import stats
from scipy.special import expit
import seaborn as sns
class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass

    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """

        # np.empty creates an empty array only. You have to replace this with your code.
        X = np.empty((0,0))
        y = np.empty((0))

        if dataset == 0:
            # Implement for the abalone dataset
            data = pd.read_csv('C:/Users/Ritik garg/Desktop/MlAssignment/Assignment1/abalone.txt',header = None)
            data = data.sample(frac = 1) 
            print(data.head())
            X=data.iloc[:,1:-1].to_numpy()
            # norm = np.linalg.norm(X)
            # X = X/norm
            y=data.iloc[:,-1].to_numpy()
            print("Features")
            print(X)
            print("Output")
            print(y)
            
            pass
        elif dataset == 1:
            # Implement for the video game dataset
            data = pd.read_csv('C:/Users/Ritik garg/Desktop/MlAssignment/Assignment1/VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv')
            # print(data.shape)
#print()    
            # sns.heatmap(data.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
            data = data.sample(frac = 1) 
            data = data[data['Critic_Score'].notna()]
            data = data[data['User_Score'].notna()]
            data = data[data['Global_Sales'].notna()]
            data['User_Score']= pd.to_numeric(data['User_Score'],errors='coerce')
            data = data[data['User_Score'].notna()]
            data = data[['Critic_Score','User_Score','Global_Sales']]
            z = np.abs(stats.zscore(data))
            data = data[(z<3).all(axis=1)]
            #normalise -> remove outliers
            print(data)
            X=data.iloc[:,:-1].to_numpy()
            # X = Normalizer().fit_transform(X)
            norm = np.linalg.norm(X)
            X = X/norm
            y=data.iloc[:,-1].to_numpy()
            # y = Normalizer().fit_transform(y)
            print("Features")
            print(X)
            print("Output")
            print(y)
            # plt.figure()
            # plt.scatter(X,y)
            # plt.title("Normalized Data")
            # plt.show()
            pass
        elif dataset == 2:
            # Implement for the banknote authentication dataset
            data = pd.read_csv('C:/Users/Ritik garg/Desktop/MlAssignment/Assignment1/data_banknote_authentication.txt',header = None)
            data = data.sample(frac = 1)
            z = np.abs(stats.zscore(data))
            data = data[(z<3).all(axis=1)]
            print(data.info())
            print(data.describe())
            # print(data.quality.unique() )
            print(data.describe())

            # sns.heatmap(data.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
            
            # plt.figure(figsize=(6,4))
            # sns.heatmap(data.corr(),cmap='viridis',annot=True) 

            X=data.iloc[:,1:-1].to_numpy()
            # norm = np.linalg.norm(X)
            # X = X/norm
            # norm = np.linalg.norm(X)
            # X = X/norm
            y=data.iloc[:,-1].to_numpy()
            # y = Normalizer().fit_transform(y)
            print("Features")
            print(X)
            print("Output")
            print(y)
            pass
        elif dataset == 3:
            # Implement for the banknote authentication dataset
            data = pd.read_csv('C:/Users/Ritik garg/Desktop/MlAssignment/Assignment1/Q4_Dataset.txt',sep=" ",header = None)
            # data = data.sample(frac = 1)
            data = data.iloc[:,[4,7,11]]
            print(data)
            # z = np.abs(stats.zscore(data))
            # data = data[(z<3).all(axis=1)]
            # print(data.info())
            # print(data.describe())
            # print(data.quality.unique() )
            # print(data.describe())

            # sns.heatmap(data.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
            
            # plt.figure(figsize=(6,4))
            # sns.heatmap(data.corr(),cmap='viridis',annot=True)
            X=data.iloc[:,[1,2]].to_numpy()
            print(X.shape)
            # X=data.iloc[:,1:].to_numpy()
            # norm = np.linalg.norm(X)
            # X = X/norm
            # norm = np.linalg.norm(X)
            # X = X/norm
            y=data.iloc[:,[0]].to_numpy()
            print(y.shape)
            # y = Normalizer().fit_transform(y)
            print("Features")
            print(X)
            print("Output")
            print(y)
            pass

        return X, y

class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        self.__X = X
        # self.__y = y.reshape(-1,1)
        self.__y = y
        # self.X_features = X.shape[1]
        # data = pd.read_csv('C:/Users/Ritik garg/Desktop/MlAssignment/Assignment1/VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv')
        # print(data.shape)
        # #print()
        # #data.plot(kind = "scatter",x = 'Critic_Score',y='Global_Sales')
        # data.plot(kind = "scatter",x = 'User_Score',y='Global_Sales')
        
        # plt.show()	
        #plt.scatter(X,y)
        print(X.shape)
        print(y.shape)
        self.coefficient = np.random.randn(X.shape[1]);  #created an array of size 2 with random values for the coefficients
        self.intercept =np.random.random();  #Created a random value for the bias
        # print(self.coefficient)        
        RMSE_errors = []
        Rmse_coef = []
        Rmse_intercept = []
        # errors = []
        for i in range(2000):
            self.RMSE_gradientDescent()
            Rmse_coef.append(self.coefficient)
            Rmse_intercept.append(self.intercept)
            RMSE_errors.append(self.RMSE_errors())
          # print(self.coefficient,self.intercept)
          # return errors
        self.coefficient = np.random.randn(X.shape[1]);  #created an array of size 2 with random values for the coefficients
        self.intercept =np.random.random();  #Created a random value for the bias
        # print(self.coefficient)        
        print("RMSE_errors-> " + str(self.RMSE_errors()))
        MAE_errors = []
        Mae_coef = []
        Mae_intercept = []
        # errors = []
        for i in range(2000):
            self.MAE_gradient_descent()
            Mae_coef.append(self.coefficient)
            Mae_intercept.append(self.intercept)
            MAE_errors.append(self.MAE_errors())
        # plt.plot(kinf = 'scatter',x=errors,y=self.__y)
        # return self
        print("MAE Errors-> " + str(MAE_errors[-1]))
        # print("stochastic_errors-> " + str(stochastic_errors[-1]))
        # print("RMSE coefficient -> ")
        return RMSE_errors, MAE_errors, Rmse_coef, Rmse_intercept, Mae_coef, Mae_intercept

    def RMSE_gradientDescent(self):  
      d_coefficient,d_intercept = self.RMSE_gradient()
      coef = []
      intercept = []
      learningRate = 0.01
      self.coefficient = self.coefficient - d_coefficient*learningRate
      self.intercept = self.intercept - d_intercept*learningRate
      coef.append(self.coefficient)
      intercept.append(self.intercept)
      # return coef,intercept

    def RMSE_gradient(self):
      yHat = self.predict(self.__X)     #yHat = mx + c
      cost = (((yHat - self.__y) ** 2).sum()/self.__X.shape[0])**(0.5)
      # these part can be done using loops but will take more time to compute, so using inbuilt functions

      d_coefficient  = np.dot((yHat - self.__y).T, self.__X)/(self.__X.shape[0] * cost)
      d_intercept = (yHat - self.__y).mean()/cost   #intercept is the sum of yHat - y divided by number of enteries i.e the mean

      return d_coefficient, d_intercept

    def MAE_gradient_descent(self):
      d_coefficient,d_intercept = self.MAE_gradient()
      learningRate = 0.01
      coef=[]
      intercept = []
      self.coefficient = self.coefficient - d_coefficient*learningRate
      self.intercept = self.intercept - d_intercept*learningRate
      coef.append(self.coefficient)
      intercept.append(self.intercept)
      # return coef, intercept

    def MAE_gradient(self):
      yHat = self.predict(self.__X)     #yHat = mx + c
      # cost = ((np.abs(yHat - self.__y)).sum()/self.__X.shape[0])
      # these part can be done using loops but will take more time to compute, so using inbuilt functions
      cost = (yHat-self.__y)
      cost = (cost)/np.abs(cost)
      d_coefficient  = np.dot(cost.T, self.__X)/(self.__X.shape[0])
      # d_coefficient = abs(d_coefficient)/d_coefficient
      d_intercept = (cost).mean()   #intercept is the sum of yHat - y divided by number of enteries i.e the mean
      # d_intercept = abs(d_intercept)/d_intercept
      # if(cost<0):
      #   d_coefficient = (-1)*d_coefficient
      #   d_intercept = (-1)*d_intercept
      return d_coefficient, d_intercept

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        # return np.sum((X*self.coefficient),axis=1,keepdims = True) + self.intercept
        return X@self.coefficient + self.intercept 

    def RMSE_errors(self):
      return (((self.predict(self.__X) - self.__y) ** 2).sum()/self.__X.shape[0])**(0.5)

    def MAE_errors(self):
        return (abs(self.predict(self.__X)-self.__y).sum()/self.__X.shape[0])

    # def predictUsingNormalEquation(self,X,y):
    #     ones=np.ones(X.shape)
    #     X=np.append(ones,X,axis=1)
    #     inv=np.linalg.inv(np.matmul(self.__X.T,self.__X))
    #     self.theta=np.matmul(np.matmul(inv,self.__X.T),self.__y)
    #     y_pred=np.matmul(X,self.theta)
    #     return y_pred,(abs(y-y_pred)/y)*100

    def validationLoss(self,X,y,Rmse_coef,Rmse_intercept,Mae_coef,Mae_intercept):
        RMSE_errors = []
        MAE_errors = []
        # print("X[0]" + str(X[0]))
        # print(Rmse_coef[0])
        # print(Rmse_intercept[0])
        for i in range(len(Rmse_intercept)):
            y_predicted = X@Rmse_coef[i] + Rmse_intercept[i]
            RMSE_errors.append((((y-y_predicted)**2).sum()/y[0])**(0.5))
            # MAE_errors.append(abs(y_actual - y_predected).mean())
        for i in range(len(Mae_intercept)):
            y_predicted = X@Mae_coef[i] + Mae_intercept[i]
            MAE_errors.append(abs(y - y_predicted).mean())
        return RMSE_errors,MAE_errors


class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        self.__X = X
        self.__y = y

        self.coefficient = np.random.randn(X.shape[1])
        self.intercept = np.random.random()
        Batch_errors = []
        stochastic_errors = []
        Batch_coef = []
        Batch_intercept = []
        for i in range(10000):
            self.gradient_descent()
            Batch_coef.append(self.coefficient)
            Batch_intercept.append(self.intercept)
            Batch_errors.append(self.error())
        print("After Batch Gradient Descent")
        print(self.coefficient, self.intercept)
        print("Accuracy on train after Batch Gradient Descent: " + str(self.accuracy(X,y,self.coefficient,self.intercept)))
        self.coefficient = np.random.randn(X.shape[1])
        self.intercept = np.random.random()
        Stochastic_coef = []
        Stochastic_intercept = []
        for i in range(10000):
            self.stochastic_gradient_descent()
            Stochastic_coef.append(self.coefficient)
            Stochastic_intercept.append(self.intercept)
            stochastic_errors.append(self.error())
            # print("error-> " + str(stochastic_errors[i]))
          # print(self.coefficient,self.intercept)
          # return errors
        # plt.plot(kinf = 'scatter',x=errors,y=self.__y)
        # return self
        print("Accuracy on train Stochastic Gradient Descent: " + str(self.accuracy(X,y,self.coefficient,self.intercept)))
        print("After Stochastic Gradient Descent")
        print(self.coefficient, self.intercept)

        print("Batch Errors-> " + str(Batch_errors[-1]))
        print("stochastic_errors-> " + str(stochastic_errors[-1]))
        return Batch_errors,stochastic_errors, Batch_coef, Batch_intercept, Stochastic_coef, Stochastic_intercept

    def gradient_descent(self):
    	d_coefficient, d_intercept = self.gradient()
    	learningRate = 10
    	self.coefficient = self.coefficient - d_coefficient*learningRate
    	self.intercept = self.intercept - d_intercept*learningRate

    def gradient(self):
        yHat = self.predict(self.__X)
        d_coefficient = 0.0
        d_intercept = 0.0
        # for i in range(self.__X.shape[0]):
        #     d_coefficient += (np.dot((yHat[i] - self.__y[i]).T,self.__X[i]))
        #     d_intercept += (yHat[i] - self.__y[i])
        # d_coefficient /=self.__X.shape[0]
        # d_intercept /= self.__X.shape[0]
        d_coefficient = np.dot((yHat-self.__y).T, self.__X).flatten() / self.__X.shape[0]
        d_intercept = (yHat-self.__y).mean()
        return d_coefficient,d_intercept

    def stochastic_gradient_descent(self):
        learningRate = 10
        d_coefficient = 0.0
        d_intercept = 0.0
        # t=10
        # for i in range(self.__X.shape[0]):
        # #     # print(self.__X[i])
        #     t = np.random.randint(low=1, high=self.__X.shape[0]-1)
        #     d_coefficient,d_intercept = self.stochastic_gradient(t)
        i = np.random.randint(low=0, high=self.__X.shape[0])  #1000
        # for i in range(self.__X.shape[0]):
        d_coefficient,d_intercept = self.stochastic_gradient(i)
        self.coefficient = self.coefficient -  learningRate*d_coefficient
        self.intercept = self.intercept - learningRate*d_intercept
        # print("predicted coefficient and intercept -> ")
        # print(self.coefficient,self.intercept)

    def stochastic_gradient(self,i):
        yHat = self.predict(self.__X)
        # print("yHat->" + str(yHat))
        # d_coefficient = 0.0   #they must be defined before use
        # d_intercept = 0.0
        d_coefficient = np.dot((yHat[i] - self.__y[i]).T,self.__X[i]).flatten()
        d_intercept = (yHat[i] - self.__y[i])
        # print(d_coefficient,d_intercept, i)
        return d_coefficient,d_intercept

    def predict(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        # print(self.coefficient,self.intercept)
        g = X.dot(self.coefficient) + self.intercept
        yhat = self.signoid(g)
        # print("yHat -> ")
        return yhat

    def signoid(self,g):
        # if(g.any()>=0):
        #     y = np.exp(-g)
        #     return 1/(1+y)
        # else:
        #     ''' if g is less than 0, then y will be small and denominator cannot be zero because it is 1 + exp(y)'''
        #     y = np.exp(g)
        #     return y/(1+y)
        y = 1.0/(1.0+np.exp(-1.0*g))
        return y
        # return expit(g)


    def error(self):
    	yHat = self.predict(self.__X)
    	# print((((-1)*((self.__y*np.log2(yHat)) + ((1-self.__y)*np.log2(1-yHat))).sum())/self.__X.shape[0]))
    	return ((-1.0)*((self.__y*np.log(yHat+1e6)) + ((1.0-self.__y)*np.log(1.0-yHat+1e6))).mean())

    def accuracy(self,X,y,coef,intercept):
    	# yHat = self.predict(self.__X)
    	# return np.abs((yHat - self.__y)).sum()/self.__X.shape[0]
        g = X.dot(coef) + intercept
        yhat = self.signoid(g)
        ypredicted = yhat.flatten()
        ypredicted = np.round(ypredicted)
    	# print(ypredicted == y)
        return (ypredicted == y).mean()

    def validationLoss(self,X,y,Batch_coef,Batch_intercept,Stochastic_coef,Stochastic_intercept):
        Batch_errors = []
        stochastic_errors = []
        for i in  range(len(Batch_intercept)):
            g = X.dot(Batch_coef[i]) + Batch_intercept[i]
            yhat = self.signoid(g)
            Batch_errors.append((-1.0)*((y*np.log(yhat)) + ((1.0-y)*np.log(1.0-yhat))).mean())
        for i in range(len(Stochastic_intercept)):
            g = X.dot(Stochastic_coef[i]) + Stochastic_intercept[i]
            yhat = self.signoid(g)
            stochastic_errors.append((-1.0)*((y*np.log(yhat)) + ((1.0-y)*np.log(1.0-yhat))).mean())

        return Batch_errors,stochastic_errors


