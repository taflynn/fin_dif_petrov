# PACKAGES
# Fix the number of threads per process to be 1
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from mpi4py import MPI
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# ABSOLUTE PATH (might need to change using pwd) 
os.chdir('/home/b6019832/Documents/mu_finder')
# Petrov functions
from petrov_im_tim_rk4 import petrov_im_tim_rk4_mat # RK4 imaginary time function
from petrov_real_tim_rk4 import petrov_real_tim_rk4_mat # RK4 real time function
from freq_funcs import curve_fitting # curve fitting function

# Setup MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# SWITCHES:
# trap? 0=NO,1=YES
trap = 0
# interacting gas? (0 = Non-interacting gas, 1 = Interacting gas w/ LHY)
int_gas = 1
# breathing mode? (0 = No initial phase perturbation, hence no oscillation, 1 = Small phase perturbation)
mode = 1

# PARAMETERS (where the N values are (N-N_c)^(1/4) as given in Petrov 2015)
N_max = 7.5 # maximum rescaled atom number
N_min = 1 # minimum rescaled atom number
N_steps = 16 # number of values of N per process

# INITIALISE ARRAYS FOR SCATTERING AND GATHERING
N_tilde = None 
Mu = None
Omega = None

# N_ARRAY SETUP 
if rank == 0:
    N_tilde = np.linspace(N_min,N_max,size*N_steps) # atom number array
    Mu = np.empty(len(N_tilde)) # empty Mu array for gathering
    Omega = np.empty(len(N_tilde)) # empty breathing mode freq array for gathering
    Nsize = len(N_tilde) # size of N_array
else:
    Nsize = None # size of N_array not named specified on root process
Nsize = comm.bcast(Nsize,root=0) # broadcast the size of atom number array

# SCATTER THE N ARRAY ACROSS THE PROCESSES
N_partial = np.empty(Nsize//size).astype(float) # setting a smaller array to receive scattered atom number array
comm.Scatter(N_tilde,N_partial,root=0) # scatter atom number array
print("from process ",rank," N_partial is = ",N_partial) # just some terminal output to show atom numbers considered

# MU_ARRAY SETUP
mu_array = np.empty(len(N_partial)).astype(float) # empty smaller mu array to store mu for scattered atom numbers
omega_array = np.empty(len(N_partial)).astype(float) # likewise for breathing mode freq values

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

# POTENTIAL
if trap == 0:
    V = np.zeros(r.size)
elif trap == 1:
    V = 0.5*r**2

# INITIALISE WAVEFUNCTION
phi_0 = np.exp(-(r)**2/(2*(2)**2)) # Gaussian initial condition (do not set width = 1 or width too large)
Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm) # normalised initial condition

# ITERATING ACROSS THE N PARAMETER SPACE
for i in range(0,len(N_partial)):
    N_current = N_partial[i]**4+18.65 # extracting the true rescaled atom number value
    # IMAGINARY TIME
    print("!BEGUN! process: ",rank,"has just begun the groundstate function for N = ",N_current) # some terminal output
    [phi,mu_array[i],tol_mode] = petrov_im_tim_rk4_mat(phi_0,r,dr,dt,N_current,V,int_gas,im_t_steps)
    # REAL TIME (for certain values of N)
    if N_current>1050:
        # REAL TIME FUNCTION
        [phi,spacetime,t_array,mean_r2]	= petrov_real_tim_rk4_mat(phi,mu_array[i],r,dr,dt,N_current,V,int_gas,t_steps,mode)
        # EXTRACTING FREQUENCY OF THE <r^2> OBSERVABLE
        omega_array[i] = curve_fitting(t_array,mean_r2)	
    else:
        # OTHERWISE SETTING THE BREATHING MODE TO BE NaN FOR PLOTTING PURPOSES
        omega_array[i] = np.NaN 
    print("!COMPLETED! process: ",rank," just completed N = ",N_current,", with mu = ",mu_array[i]," and density tol = ",tol_mode) # some terminal output

# Gather together the mu's from each process and save them into a large mu array
comm.Gather(mu_array,Mu,root=0) # gathered mu values
comm.Gather(omega_array,Omega,root=0) # gathered breathing mode freq values

# SAVE FIGURE ON THE ROOT PROCESS
if comm.rank == 0:
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
