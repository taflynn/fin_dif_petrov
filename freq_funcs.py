import numpy as np
from scipy.optimize import curve_fit

def curve_fitting(t_array,mean_r2):
	def sine_func(x,a,b,c,d,f):
		return a*np.exp(-f*x)*np.sin(b*x + c) + d
	A = max(mean_r2) # initial guess of amplitude
	B = 0.5 # very rough guess for frequency
	C = 1 # default input value for phase shift
	D = mean_r2[0] # initial guess for the y shift of the sine curve
	popt = curve_fit(sine_func,t_array[0:-1],mean_r2[0:-1],p0=[A,B,C,D,0.01])[0]
	return popt[1]
