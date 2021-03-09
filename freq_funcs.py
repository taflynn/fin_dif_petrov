import numpy as np
from scipy.optimize import curve_fit

# CURVE FITTING FUNCTION TO FIT BREATHING MODE DATA TO SINE WAVE
def curve_fitting(t_array,mean_r2):
    # Define a generic damped sine curve
    def sine_func(x,a,b,c,d,f):
        return a*np.exp(-f*x)*np.sin(b*x + c) + d
    
    # Initial guess parameters for the curve_fit function
    A = max(mean_r2) # initial guess of amplitude
    B = 0.5 # very rough guess for frequency
    C = 1 # default input value for phase shift
    D = mean_r2[0] # initial guess for the y shift of the sine curve
    F = 0.01
    
    # Extracting the fitted parameter values from the curve_fit function
    popt = curve_fit(sine_func,t_array[0:-1],mean_r2[0:-1],p0=[A,B,C,D,F])[0]
    # Return only the second parameter as this is the breathing mode frequency
    return popt[1]
