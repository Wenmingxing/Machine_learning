#! /usr/bin/env python

import numpy as np

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
