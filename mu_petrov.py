# PACKAGES
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from mpi4py import MPI
import scipy.fftpack
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
# ABSOLUTE PATH 
os.chdir('/home/b6019832/Documents/mu_finder')
# IMAGINARY TIME FUNCTION
from petrov_im_tim_rk4 import petrov_im_tim_rk4_mat # RK4 w/ matrices in KE term
from petrov_real_tim_rk4 import petrov_real_tim_rk4_mat
from freq_funcs import curve_fitting
import pandas as pd

# Setup MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# SWITCHES:
# trap? 0=NO,1=YES
trap = 0
# interacting gas? (0 = Non-interacting gas, 1 = Interacting gas w/ LHY)
int_gas = 1

N_max = 7.5
N_min = 1
N_steps = 16
N_tilde = None
Mu = None
Omega = None
# N_ARRAY SETUP 
if rank == 0:
    N_tilde = np.linspace(N_min,N_max,size*N_steps)
    Mu = np.empty(len(N_tilde))
    Omega = np.empty(len(N_tilde))
    Nsize = len(N_tilde)
else:
    Nsize = None
Nsize = comm.bcast(Nsize,root=0)
# SCATTER THE N ARRAY ACROSS THE PROCESSES
N_partial = np.empty(Nsize//size).astype(float)
comm.Scatter(N_tilde,N_partial,root=0)
print("from process ",rank," N_partial is = ",N_partial)
# MU_ARRAY SETUP
mu_array = np.empty(len(N_partial)).astype(float)
omega_array = np.empty(len(N_partial)).astype(float)
# GRID
Lr = 32 # box length
Nr = 256 # grid points
dr = Lr/Nr # spatial step
r = np.arange(-1/2,(Nr + 3/2),1)*dr # position array with 4 ghost points

# TIME SETUP
dt = 0.1*dr**2 # time step 
im_t_steps = 250000 # number of imaginary time steps
t_steps = 30000 

# PARAMETERS
pi = np.math.pi

# POTENTIAL
if trap == 0:
    V = np.zeros(r.size)
elif trap == 1:
    V = 0.5*r**2

mode = 1

# INITIALISE WAVEFUNCTION
phi_0 = np.exp(-(r)**2/(2*(2)**2)) # Gaussian initial condition
Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm) # normalised initial condition

for i in range(0,len(N_partial)):
    N_current = N_partial[i]**4+18.65
    # IMAGINARY TIME
    print("!BEGUN! process: ",rank,"has just begun the groundstate function for N = ",N_current)
    [phi,mu_array[i],tol_mode] = petrov_im_tim_rk4_mat(phi_0,r,dr,dt,N_current,V,int_gas,im_t_steps)
    if N_current>1050:
        [phi,spacetime,t_array,mean_r2]	= petrov_real_tim_rk4_mat(phi,mu_array[i],r,dr,dt,N_current,V,int_gas,t_steps,mode)
        omega_array[i] = curve_fitting(t_array,mean_r2)	
    else:
        omega_array[i] = np.NaN 
    print("!COMPLETED! process: ",rank," just completed N = ",N_current,", with mu = ",mu_array[i]," and density tol = ",tol_mode)

# Gather together the mu's from each process and save them into a large mu array
comm.Gather(mu_array,Mu,root=0)
comm.Gather(omega_array,Omega,root=0)
if comm.rank == 0:
    DataOut = np.column_stack((N_tilde,Mu))
    # np.savetxt('mu_N_steps'+str(N_steps)+'.csv',DataOut,delimiter=',',fmt='%18.16f')
    plt.plot(N_tilde,Omega,N_tilde,-Mu)
    plt.xlim(N_tilde[0],N_tilde[-1])
    plt.ylim(-Mu[0],-Mu[-1])
    plt.xlabel("$(N - N_c)^{(1/4)}$")
    plt.legend(("$\omega_0$","-$\mu$"))
    plt.savefig("mu_w_breath_damped.png",dpi=300)
    plt.show
    
# SAVING DATA (Produces two csv files: 1) mu(N); 2) omega_0(N))
if comm.rank == 0:
    mu_data = np.column_stack((N_tilde,-Mu))
    np.savetxt('mu_petrov.csv',mu_data,delimiter=',',fmt='%18.16f')
    omega_data = np.column_stack((N_tilde,Omega))
    np.savetxt('omega0_petrov.csv',omega_data,delimiter=',',fmt='%18.16f')


# Saving data using pandas dataframe
#mu_data = {'N_tilde':N_tilde, 'Mu':-Mu}
#data_frame = pd.DataFrame(mu_data, columns = ['N_tilde', 'Mu'])
#data_frame.to_csv('mu_petrov.csv',na_rep=NULL,index=False,index_label=False)

#omega_data = {'N_tilde':N_tilde, 'Omega_0':Omega}
#data_frame = pd.DataFrame(mu_data, columns = ['N_tilde', 'Omega_0'])
#data_frame.to_csv('omega0_petrov.csv',na_rep=NULL,index=False,index_label=False)

