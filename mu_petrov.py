# PACKAGES
from mpi4py import MPI
import os
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import time
# ABSOLUTE PATH 
os.chdir('C:\\Users\\TAFly\\Documents\\PhD\\1st_year\\Python_GPE\\fin_dif_codes')
# IMAGINARY TIME FUNCTION
from petrov_im_tim_euler import petrov_im_tim_euler # euler
from petrov_im_tim_rk4 import petrov_im_tim_rk4 # RK4 w/ for loops in KE term
from petrov_im_tim_rk4 import petrov_im_tim_rk4_mat # RK4 w/ matrices in KE term

# Setup MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# start timer
tic = time.perf_counter()
    
# SWITCHES:
# trap? 0=NO,1=YES
trap = 0
# interacting gas? (0 = Non-interacting gas, 1 = Interacting gas w/ LHY)
int_gas = 1

N_max = 3145

# N_ARRAY SETUP 
if rank == 0:
    N = np.arange(1,N_max+1,100)
    Mu = np.empty(N.shape[0])
# SCATTER THE N ARRAY ACROSS THE PROCESSES
N_partial = np.empty(N_max//size).astype(int)
comm.Scatter(N,N_partial,root=0)
# MU_ARRAY SETUP
mu_array = np.empty(N_partial.shape[0]).astype(float)

# GRID
Lr = 12 # box length
Nr = 256 # grid points
dr = Lr/Nr # spatial step
r = np.arange(-1/2,(Nr + 3/2),1)*dr # position array with 4 ghost points

# TIME SETUP
dt = 0.1*dr**2 # time step 
im_t_steps = 1000000 # number of imaginary time steps

# PARAMETERS
pi = np.math.pi

# POTENTIAL
if trap == 0:
    V = 0
elif trap == 1:
    V = 0.5*r**2

for i in range(0,N_partial.shape[0]):
    # INITIALISE WAVEFUNCTION
    phi_0 = np.exp(-(r)**2/(2*(1)**2)) # Gaussian initial condition
    Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
    phi_0 = phi_0/np.sqrt(Norm) # normalised initial condition
    phi = phi_0
    # IMAGINARY TIME
    [phi,mu_array[i],tol_mu,tol_mode] = petrov_im_tim_rk4_mat(phi_0,r,dr,dt,N_partial[i],V,int_gas,im_t_steps)

comm.Gather(mu_array,mu,root=0)

if comm.rank == 0:
    plt.plot((N-18.65)**(0.45),Mu)
    plt.xlabel("$(N - N_{c})^{1/4}$")
    plt.ylabel("\mu")
    plt.savefig("mu_parallel.png",dpi=300)
    plt.show
    
