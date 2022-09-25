import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.linalg import pinv
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import SGDRegressor, LinearRegression


class LinearRegressor:
    def __init__(self, epochs=300, learning_rate=0.0001, solver='batch', batch_size=20, regularization='elastic', l1=0.7, l2=0.3, lambda_=0):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.solver = solver
        self.batch_size = batch_size
        self.regularization = regularization
        self.l1 = l1
        self.l2 = l2
        self.lambda_ = lambda_

    def prepareData(self, X, y):
        X = np.c_[np.ones([X.shape[0], 1]), X]
        y = y.reshape(X.shape[0], 1)
        return X,y

    def fit(self, X, y):
        X,y = self.prepareData(X, y)
        self.W = np.random.randn(X.shape[1], 1)

        if self.solver == 'batch':
            self.alias = "Batch Gradient Descent"
            self.batchGradientDescent(X, y)
        elif self.solver == 'mini_batch':
            self.alias = "Mini Batch Gradient Descent"
            self.miniBatchGradientDescent(X, y)
        elif self.solver == 'stochastic':
            self.alias = "Stochastic Gradient Descent"
            self.batch_size = 1
            self.miniBatchGradientDescent(X, y)
        elif self.solver == "SVD":
            self.alias = "Singular Value Decomposition"
            self.svd(X, y) 
        else:
            self.alias = "Normal Equation"
            self.normalEquation(X,y)
        
        return self

    def batchGradientDescent(self, X, y):
        for i in tqdm(range(self.epochs), desc="Fitting a batch gradient descent model into the data..."):
            self.updateGradients(X, y)
        return self

    def miniBatchGradientDescent(self, X, y):
        model_name = "stochastic gradient descent" if self.solver == "stochastic" else "mini-batch gradient descent"
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]
        for epoch in tqdm(range(self.epochs), desc=f"Fitting a {model_name} model into the data..."):
            step = 1 if self.solver == "stochastic" else self.batch_size
            for j in range(0, X.shape[0], step):
                Xi = X[j:j+step]
                yi = y[j:j+step]
                self.updateGradients(Xi, yi)

    def normalEquation(self, X, y):
        XTX = np.matmul(X.T,X)
        XTX_inv = np.linalg.inv(XTX)
        XTX_invXT = np.matmul(XTX_inv,X.T)
        self.W = np.matmul(XTX_invXT,y)
    
    def svd(self, X, y):
        U, S, VT = np.linalg.svd(X, full_matrices=0)
        self.W = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

    def updateGradients(self, X, y):
        y_pred = X.dot(self.W)
        dw = (1/X.shape[0]) * X.T.dot(y_pred - y)
        if self.regularization:
            dw += self.getRegularization(type_=self.regularization, W = self.W, n=X.shape[1])
        self.W = self.W -  2 * self.learning_rate * dw

    def predict(self, X):
        X = np.c_[np.ones([X.shape[0], 1]), X]
        return X.dot(self.W)

    def getRegularization(self, type_, W, n):
        if type_ == 'elastic':
            ratios_sum = self.l1 + self.l2
            if ratios_sum == 1:
                return self.lambda_ * (self.l1 * np.sign(W) + self.l2 * W)
            else:
                return self.lambda_ * (self.l1/ratios_sum * np.sign(W) + self.l2/ratios_sum * 2 * np.sum(W))
        elif type_ == 'l1':
            return self.l1 * np.sign(W)
        elif type_ == 'l2':
            return self.l2 * np.sum(W)
        else:
            return 0


if __name__ == "__main__":
    np.random.seed(42)
    dataframe = load_diabetes()
    dataframe = pd.DataFrame(data= np.c_[dataframe['data'], dataframe['target']],
                        columns= dataframe['feature_names'] + ['target'])

    dataframe.dropna(inplace=True)

    X = dataframe.drop('target', axis=1).values
    y = dataframe['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    attr_scaler = StandardScaler()
    X_train = attr_scaler.fit_transform(X_train)
    X_test = attr_scaler.transform(X_test)

    sklearn_sgd_model = SGDRegressor()
    sklearn_sgd_model.fit(X_train,y_train)
    sklearn_y_hat = sklearn_sgd_model.predict(X_test)
    print("Sklearn SGD model MAE ==>", mae(y_test, sklearn_y_hat))

    sklearn_linear_model = LinearRegression()
    sklearn_linear_model.fit(X_train, y_train)
    sklearn_linear_y_hat = sklearn_linear_model.predict(X_test)
    print("Sklearn Linear Regression model MAE ==>", mae(y_test, sklearn_linear_y_hat))
