#! /usr/bin/env python


#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
#df.tail()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#! /usr/bin/env python

#import numpy as np

class Perceptron(object):
	""" Perceptron classifier.
	parameters
	-----------
	eta:float
		learning rate (between 0.0 and 1.0)
	n_iter:int
		passes over the training dataset
	
	Attributes
	------------
	w_:1d_array
		weights after fitting
	errors_:list
		number of misclassifications in every epoch.
	"""
	def __init__(self,eta = 0.01, n_iter=10):
		self.eta= eta
		self.n_iter = n_iter

	def fit(self,x,y):
		''' fit training data.

	parameters
	-----------
	x: {array-like},shape=[n_samples,n_features]
		Training vectors,where n_samples is the number of samples
		and n_features is the number of features.
	y: array_like,shape = [n_samples]
		target values

	returns
	-----------
	self.object
		'''

		self.w_ = np.zeros(1 + x.shape[1])# Add w_0
		self.errors_ = []
		for _ in range(self.n_iter):
			errors = 0
			for xi,target in zip(x,y):
				update = self.eta*(target - self.predict(xi))
				self.w_[1:] += update*xi
				self.w_[0] += update
				errors += int(update!=0)
			self.errors_.append(errors)
		return self


	def net_input(self,x):
		'''calculate net input '''
		return np.dot(x,self.w_[1:])+self.w_[0]
	def predict(self,x):
		'''return class label after unit step'''
		return np.where(self.net_input(x) >= 0.0,1,-1) #analogue ?: in c ++

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

y = df.iloc[0:100,4].values

y = np.where(y == 'Iris-setosa',-1,1)

x = df.iloc[0:100,[0,2]].values

'''plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
'''

ppn = Perceptron(eta=0.1,n_iter=10)

ppn.fit(x,y)
'''
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker = '0')
plt.xlabel('Epoches')
plt.ylabel('Number of misclassfications')
plt.show()
'''

from matplotlib.colors import ListedColormap

def plot_decision_region(x,y,classifier,resolution=0.02):
	#setup marker generation and color map
	markers = ('s','x','o','^','v')
	colors = ('red','blue','lightgreen','gray','cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
		
	# plot the secision surface
	x1_min,x1_max = x[:,0].min()-1,x[:,0].max()+1
	x2_min,x2_max = x[:,1].min()-1,x[:,1].max()+1

	xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
	z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
	z = z.reshape(xx1.shape)
	
	plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
	plt.xlim(xx1.min(),xx1.max())
	plt.ylim(xx2.min(),xx2.max())

	# plot clas samples

	for idx,cl in enumerate(np.unique(y)):
		plt.scatter(x=x[y==cl,0],y=x[y==cl,1],alpha=0.8,c=cmap(idx),marker = markers[idx],label=cl)
	

plot_decision_region(x,y,classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.show()
