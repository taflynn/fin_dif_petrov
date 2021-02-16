import numpy as np
from scipy.optimize import curve_fit

def sine_func(x,a,b,c,d):
	return a*np.sin(b*x + c) + d

def curve_fitting(t_array,mean_r2):
	A = max(mean_r2) # initial guess of amplitude
	B = 0.5 # very rough guess for frequency
	C = 1 # default input value for phase shift
	D = mean_r2[0] # initial guess for the y shift of the sine curve
	popt = curve_fit(sine_func,t_array[0:-1],mean_r2[0:-1],p0=[A,B,C,D])[0]
	return popt[1]
