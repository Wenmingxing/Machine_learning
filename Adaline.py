#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

ada1 = AdalineGD(n_iter=10,eta=0.01).fit(x,y)

ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
	
ax[0].set_xlabel('Epoches')
ax[0].set_ylabel('log(sum-squared-error)')
ax[0].set_title('Adalie-learning rate 0.01')

ada2 = AdalineGD(n_iter=10,eta=0.0001).fit(x,y)
	
ax[1].plot(range(1,len(ada2.cost_)+1),ada2.cost_,marker='o')
ax[1].set_xlabel('Epoches')
ax[1].set_ylabel('sum-sqaured-error')
ax[1].set_title('Adaline-learning rate 0.001')
plt.show()			
