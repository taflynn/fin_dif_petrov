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

# Setup MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# SWITCHES:
# trap? 0=NO,1=YES
trap = 0
# interacting gas? (0 = Non-interacting gas, 1 = Interacting gas w/ LHY)
int_gas = 1

N_max = 7
N_min = 1
N_steps = 6
N_tilde = None
Mu = None
# N_ARRAY SETUP 
if rank == 0:
    N_tilde = np.linspace(N_min,N_max,size*N_steps)
    Mu = np.empty(len(N_tilde))
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

# GRID
Lr = 24 # box length
Nr = 256 # grid points
dr = Lr/Nr # spatial step
r = np.arange(-1/2,(Nr + 3/2),1)*dr # position array with 4 ghost points

# TIME SETUP
dt = 0.1*dr**2 # time step 
im_t_steps = 250000 # number of imaginary time steps

# PARAMETERS
pi = np.math.pi

# POTENTIAL
if trap == 0:
    V = np.zeros(r.size)
elif trap == 1:
    V = 0.5*r**2

# INITIALISE WAVEFUNCTION
phi_0 = np.exp(-(r)**2/(2*(2)**2)) # Gaussian initial condition
Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm) # normalised initial condition

for i in range(0,len(N_partial)):
    N_current = N_partial[i]**4+18.65
    # IMAGINARY TIME
    print("!BEGUN! process: ",rank,"has just begun the groundstate function for N = ",N_current)
    [phi,mu_array[i],tol_mu,tol_mode,t] = petrov_im_tim_rk4_mat(phi_0,r,dr,dt,N_current,V,int_gas,im_t_steps)
    print("!COMPLETED! process: ",rank," just completed N = ",N_current,", with mu = ",mu_array[i]," and tol = ",tol_mu)

# Gather together the mu's from each process and save them into a large mu array
comm.Gather(mu_array,Mu,root=0)

if comm.rank == 0:
    DataOut = np.column_stack((N_tilde,Mu))
    # np.savetxt('mu_N_steps'+str(N_steps)+'.csv',DataOut,delimiter=',',fmt='%18.16f')
    plt.plot(N_tilde,-Mu)
    plt.xlim(N_tilde[0],N_tilde[-1])
    plt.ylim(-Mu[0],-Mu[-1])
    plt.xlabel("$(N - N_c)^{(1/4)}$")
    plt.ylabel("$\mu$")
    plt.savefig("mu_parallel.png",dpi=300)
    plt.show
    
