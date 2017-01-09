#! /usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
sys.path.append('~/ML/plot_decision.py')
from plot_decision import plot_decision_region

class AdalineGD(object):
	
	
	def __init__(self,eta=0.01,n_iter=50):
		self.eta = eta
		self.n_iter = n_iter
	
	def fit(self,x,y):
		
		self.w_ = np.zeros(1+x.shape[1])
		self.cost_ = []
		
		for i in range(self.n_iter):
			output = self.net_input(x)
			errors = (y-output)
			self.w_[1:] += self.eta*x.T.dot(errors)
			self.w_[0] += self.eta*errors.sum()
			cost = (errors**2).sum()/2.0
			self.cost_.append(cost)
		return self
	
	def net_input(self,x):
		'''calculate net input'''
		return np.dot(x,self.w_[1:]) + self.w_[0]
	def activation(self,x):
	#compute linear activation
		return self.net_input(x)
	def predict(self,x):
		#return class label after unit step
		return np.where(self.activation(x) >= 0.0,1,-1)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

y = df.iloc[0:100,4].values

y = np.where(y == 'Iris-setosa',-1,1)

x = df.iloc[0:100,[0,2]].values

x_std = np.copy(x)
x_std[:,0] = (x[:,0]-x[:,0].mean())/x[:,0].std()
x_std[:,1] = (x[:,1]-x[:,1].mean())/x[:,1].std()


ada = AdalineGD(n_iter=15,eta=0.01)
ada.fit(x_std,y)
plot_decision_region(x_std,y,classifier=ada)

plt.figure(1)
plt.title('Adaline - gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')


plt.figure(2)
plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('epoches')
plt.ylabel('sum-squared-error')
plt.show()
