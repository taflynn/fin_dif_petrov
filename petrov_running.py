# PACKAGES
# Fix the number of threads per process to be 1
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import eye
import matplotlib.pyplot as plt
#import time
# ABSOLUTE PATH 
os.chdir('/home/thomasflynn/Documents/PhD/1st_year/Petrov_GPE/fin_dif_codes')# PACKAGES
from petrov_im_tim_rk4 import petrov_im_tim_rk4_mat # RK4 imaginary time function
from petrov_real_tim_rk4 import petrov_real_tim_rk4_mat # RK4 real time function
from freq_funcs import curve_fitting # curve fitting function

# SWITCHES:
# trap? 0=NO,1=YES
trap = 1
# interacting gas? (0 = Non-interacting gas, 1 = Interacting gas w/ LHY)
int_gas = 1
# breathing mode? (0 = No initial phase perturbation, hence no oscillation, 1 = Small phase perturbation)
mode = 1

# GRID
Lr = 32 # box length
Nr = 256 # grid points
dr = Lr/Nr # spatial step
r = np.arange(-1/2,(Nr + 3/2),1)*dr # position array with 4 ghost points

# TIME SETUP
dt = 0.1*dr**2 # time step 
im_t_steps = 250000 # number of imaginary time steps
t_steps = 30000 # number of real time steps (do not go too high as curve fitting can break down)

# PARAMETERS
pi = np.math.pi
N = 3000
a = 0.05

# POTENTIAL
if trap == 0:
    V = np.zeros(r.size)
elif trap == 1:
    V = 0.5*a*r**2

# INITIALISE WAVEFUNCTION
phi_0 = np.exp(-(r)**2/(2*(5)**2)) # Gaussian initial condition (do not set width = 1 or width too large)
Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm) # normalised initial condition
# IMAGINARY TIME
[phi,mu,tol_mode] = petrov_im_tim_rk4_mat(phi_0,r,dr,dt,N,V,int_gas,im_t_steps)

phi_ground = phi
#N = 5025
# REAL TIME
[phi,spacetime,t_array,mean_r2]	= petrov_real_tim_rk4_mat(phi,mu,r,dr,dt,N,0.99*V,int_gas,t_steps,mode)

# SAVE FIGURE ON THE ROOT PROCESS
plt.plot(t_array,mean_r2)
plt.xlim(t_array[0],t_array[-1])
plt.xlabel("$t$")
plt.ylabel("$<r^2>$")
plt.savefig("breath_N"+str(N)+"a"+str(a)+".png",dpi=300)
"""
plt.plot(r,N*np.abs(phi_ground)**2)
plt.xlim(r[0],r[-1])
plt.ylim(0,5)
plt.xlabel("$r$")
plt.ylabel("$n_0$")
plt.savefig("ground_trapped_a"+str(a)+".png",dpi=300)
plt.clf()
"""
R,T = np.meshgrid(r,t_array)
plt.figure(figsize=(12,10))
plt.pcolor(R,T,N*np.abs(spacetime.T)**2 - N*np.abs(phi_ground)**2,cmap="magma",shading="auto")
clb = plt.colorbar()
clb.ax.set_title('n(r,t)')
plt.xlabel("r")
plt.ylabel("t")
plt.title("Density of $\phi(r,t)$")
plt.savefig('spacetime_breath_N'+str(N)+'a'+str(a)+".png",dpi=300)
"""
# SAVING DATA 
obs_data = np.column_stack((t_array,mean_r2))
np.savetxt('obs_quench.csv',obs_data,delimiter=',',fmt='%18.16f')
"""
"""
ground_data = np.column_stack((r,np.sqrt(N)*phi_ground))
np.savetxt('phi0_N'+str(N)+'a'+str(a)+'.csv',ground_data,delimiter=',',fmt='%18.16f')
"""
