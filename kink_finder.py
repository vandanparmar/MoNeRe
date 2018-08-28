'''
For finding the Phase Transitions in Pareto Data.
'''
from scipy import optimize
import numpy as np


def piecewise_linear(x, x0, y0, k1, k2):
'''The intersecting lines that describes the Pareto front.'''
	return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def get_kink_point(x,y):
'''Fitting the intersecting line to the data.'''
	grad = (y[0]-y[-1])/(x[0]-x[-1])
	p,e = optimize.curve_fit(f=piecewise_linear,xdata = x,ydata = y,p0=[np.mean(x),np.mean(y),grad,grad])
	return p #x0,y0,k1,k2
