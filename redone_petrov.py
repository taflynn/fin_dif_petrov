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
os.chdir('/home/thomasflynn/Documents/PhD/1st_year/Petrov_GPE/fin_dif_codes')
# IMAGINARY TIME FUNCTIONS
#from petrov_im_tim_rk4 import petrov_im_tim_rk4_mat # RK4 w/ matrices in KE term

# start timer
#tic = time.perf_counter()

# SWITCHES:
# trap? 0=NO,1=YES
trap = 0
# interacting gas? (0 = Non-interacting gas, 1 = Interacting gas w/ LHY)
int_gas = 1

# GRID
Lr = 32 # box length
Nr = 256 # grid points
dr = Lr/Nr # spatial step
r = np.arange(-1/2,(Nr + 3/2),1)*dr # position array with 4 ghost points

# TIME SETUP
dt = 0.1*dr**2 # time step 
t_steps = 100000 # number of real time steps
im_t_steps = 250000 # number of imaginary time steps
t_frame = 10000
t_save = 100

# PARAMETERS
N = 3000 # effective atom number
pi = np.math.pi

# POTENTIAL
if trap == 0:
    V = np.zeros(r.size)
elif trap == 1:
    V = 0.5*r**2

# INITIALISE WAVEFUNCTION
phi_0 = np.exp(-(r)**2/(2*(2)**2)) # Gaussian initial condition
Norm = 4*pi*np.trapz(r**2*np.abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm) # normalised initial condition
phi = phi_0

# IMAGINARY TIME (FUNCTIONS)
#[phi,mu,tol_mode,t] = petrov_im_tim_rk4_mat(phi,r,dr,dt,N,V,int_gas,im_t_steps)

# PRINTING OUT TOLERANCES FROM IMAGINARY TIME FUNCRION
#print("tol_mode = ",tol_mode)
# SAVING DATA (2 columns, spatial array and wavefunction)
#DataOut = np.column_stack((r,np.sqrt(N)*phi))
#np.savetxt('phi0_N'+str(N)+'prime.csv',DataOut,delimiter=',',fmt='%18.16f')

###################################################################################################
# IMAGINARY/REAL TIME CODES NOT PARSED INTO FUNCTIONS

# GPE COEFFICIENTS
if int_gas == 0:
    int_coef = 0
    LHY_coef = 0
elif int_gas == 1:    
    int_coef = -3*N
    LHY_coef = (5/2)*N**(3/2)    

# INITIAL MU
mu = 0.0 # initial chemical potential
tol = 1 # initialise tolerance
count = 1 # initialise imaginary time counter

# DIFFERENTIAL OPERATORS
# first order derivative in the form of a sparse matrix (centrally defined)
Dr = (1/(2*dr))*(-1*eye(phi.size-2,phi.size,k=0,dtype=float) + eye(phi.size-2,phi.size,k=2,dtype=float))
# second order derivative in the form of a sparse matrix (centrally defined 3-point formula)
Dr2 =  (1/dr**2)*(eye(phi.size-2,phi.size,k=0,dtype=float) -2*eye(phi.size-2,phi.size,k=1,dtype=float) + eye(phi.size-2,phi.size,k=2,dtype=float))

# INITIALISING ARRAYS
H_KE = np.zeros(phi.size)
H_LHY = np.zeros(phi.size)
H_int = np.zeros(phi.size)
H_trap = np.zeros(phi.size)
KE = np.zeros(phi.size)
k1 = np.zeros(phi.size)
k2 = np.zeros(phi.size)
k3 = np.zeros(phi.size)
k4 = np.zeros(phi.size)

t = 0 # initial time 
dens_current = max(N*np.abs(phi)**2)
###############################################################################
# IMAGINARY TIME LOOP:
    
for l in range(0,im_t_steps):
    # k1 CALCULATION
    KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi # Kinetic Energy derivatives
    # HAMILTONIAN TERMS
    H_KE[1:-1] = -0.5*KE[1:-1] # KE term
    H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1])**3*phi[1:-1] # LHY term
    H_int[1:-1] = int_coef*np.abs(phi[1:-1])**2*phi[1:-1] # s-wave term
    H_trap[1:-1] = V[1:-1]*phi[1:-1] # potential term

    k1[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*phi[1:-1])

    # Neumann Boundary Conditions
    k1[0] = k1[1]
    k1[-1] = k1[-2]
    
    # k2 CALCULATION
    KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k1) + Dr2 @ k1) # Kinetic Energy derivatives
    # HAMILTONIAN TERMS
    H_KE[1:-1] = -0.5*KE[1:-1] # KE term
    H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k1[1:-1]/2)**3*(phi[1:-1] + k1[1:-1]/2) # LHY term
    H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k1[1:-1]/2)**2*(phi[1:-1] + k1[1:-1]/2) # s-wave term
    H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k1[1:-1]/2) # potential term
    
    k2[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k1[1:-1]/2))
    
    # Neumann Boundary Conditions
    k2[0] = k2[1]
    k2[-1] = k2[-2]
    
    # k3 CALCULATION
    KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k2) + Dr2 @ k2) # Kinetic Energy derivatives 
    # HAMILTONIAN TERMS 
    H_KE[1:-1] = -0.5*KE[1:-1] # KE term
    H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k2[1:-1]/2)**3*(phi[1:-1] + k2[1:-1]/2) # LHY term
    H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k2[1:-1]/2)**2*(phi[1:-1] + k2[1:-1]/2) # s-wave term
    H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k2[1:-1]/2) # potential term
   
    k3[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k2[1:-1]/2))

    # Neumann Boundary Conditions
    k3[0] = k3[1]
    k3[-1] = k3[-2]
    
    #k4 CALCULATION
    KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + ((2/r[1:-1])*(Dr @ k3) + Dr2 @ k3) # Kinetic Energy derivatives 
    # HAMILTONIAN TERMS
    H_KE[1:-1] = -0.5*KE[1:-1] # KE term
    H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k3[1:-1])**3*(phi[1:-1] + k3[1:-1]) # LHY term
    H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k3[1:-1])**2*(phi[1:-1] + k3[1:-1]) # s-wave term
    H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k3[1:-1]) # potential term
    
    k4[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k3[1:-1]))
    
    # Neumann Boundary Conditions
    k4[0] = k4[1]
    k4[-1] = k4[-2]
    
    # FINAL RUNGE-KUTTA STEP    
    phi[1:-1] = phi[1:-1] + (1./6)*(k1[1:-1] + 2*k2[1:-1] + 2*k3[1:-1] + k4[1:-1])
    
    # NEUMANN BOUNDARY CONDITIONS
    # phi(j+1) - phi(j) = 0
    phi[0] = phi[1]
    phi[-1] = phi[-2]

    # WAVEFUNCTION NORMALISED
    Norm = 4*pi*np.trapz(r**2*np.abs(phi)**2)*dr
    phi = phi/np.sqrt(Norm) 
    
    # MU AND TOLERANCE CALCULATION
    if l%(im_t_steps//10)==0:
        phi_r = np.gradient(phi,dr) # derivative of wavefunction
        mu = np.trapz(r**2*(0.5*np.abs(phi_r)**2 + V*np.abs(phi)**2 + int_coef*np.abs(phi)**4 + LHY_coef*np.abs(phi)**5))/np.trapz(r**2*np.abs(phi)**2) # integral calculation of chemical potential
        dens_prev = dens_current # iterate max density
        dens_current = max(N*np.abs(phi)**2) # current max density
        tol_dens = np.abs((dens_current - dens_prev)/dens_prev) # tolerance between successive max densities
        print("mu = ",mu," and l = ",l) # print current chemical potential
    
    # IN TIME PLOTTING
    if (l % 10000 == 0):
        #plt.plot(r,N*abs(phi)**2)
        #plt.xlim((0,Lr))
        #plt.ylim((0,2))
        #plt.xlabel("$r$")
        #plt.ylabel("$n$")
        #plt.title("$N = $"+str(N))
        #plt.pause(0.2)
        print('l = ',l)
    
    # IN TIME DATA SAVING CODE
    #if (l % 10000 == 0):
    #    with open ('N_19_dens.txt','a') as f:
    #        f.write(str(t) + ' ' + str(max(np.sqrt(N)*phi)) + '\n')
    #    print(max(np.sqrt(N)*phi))
    
    # ITERATE TIME     
    t = t + dt    
    
    # ITERATE COUNTER
    count = count + 1

# SAVING DATA (2 columns, spatial array and wavefunction)
DataOut = np.column_stack((r,np.sqrt(N)*phi))
np.savetxt('phi0_N'+str(N)+'prime.csv',DataOut,delimiter=',',fmt='%18.16f')

###############################################################################
# REAL TIME LOOP:

phi_ground = phi

print("Groundstate found with $\mu_{tol}$",tol,"dens_tol",tol_dens)

# plotting the groundstate at the extrema of the box
f, axs = plt.subplots(1,2,figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(r,N*np.abs(phi)**2)
plt.xlim((0,10*dr))
plt.ylim((1.098,1.1))
plt.xlabel("$r$")
plt.ylabel("$n$")
plt.title("$N = $"+str(N)+" t = "+str(round(t)))
plt.subplot(1,2,2)
plt.plot(r,N*np.abs(phi)**2)
plt.xlim((Lr-10*dr,Lr))
plt.ylim((0,0.001))
plt.xlabel("$r$")
plt.ylabel("$n$")
plt.title("$N = $"+str(N)+" t = "+str(round(t)))       
plt.pause(0.2)

# initialising matrices
H_KE = np.zeros(phi.size).astype(complex)
H_LHY = np.zeros(phi.size).astype(complex)
H_int = np.zeros(phi.size).astype(complex)
H_trap = np.zeros(phi.size).astype(complex)
KE = np.zeros(phi.size).astype(complex)
k1 = k1.astype(complex)
k2 = k2.astype(complex)
k3 = k3.astype(complex)
k4 = k4.astype(complex)
phi = phi.astype(complex) # set groundstate wavefunction to now be complex rather than real-valued

# INITIALISING TIME AND DATA SAVING ARRAYS
t_save = 100 # save density, phase, etc. every t_save steps
t = 0 # intial time
count = 0 # intialise counter
spacetime = np.zeros((r.size,(t_steps//t_save))).astype(complex) # array to save snapshots of wavefunction
phase = np.zeros((r.size,(t_steps//t_save))).astype(complex) # array to save snapshots of the real time phase
t_array = np.zeros((t_steps//t_save)) # array to save time stamps
mean_r2 = np.zeros((t_steps//t_save)) # observable used here <r^2> 
# swap to a smaller time step in real time
dt = 0.1*dr**2

# invoke breathing mode
#lamb = 1e-4 # small constant
#phi = np.exp(1j*lamb*r**2)*phi # small phase imprint of the form exp(i*lambda*F) where F = r^2 for breathing mode
N = 4000

for l in range(0,t_steps):  
    # k1 CALCULATION
    KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi # Kinetic Energy derivatives
    # HAMILTONIAN TERMS
    H_KE[1:-1] = -0.5*KE[1:-1] # KE term
    H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1])**3*phi[1:-1] # LHY term
    H_int[1:-1] = int_coef*np.abs(phi[1:-1])**2*phi[1:-1] # s-wave term
    H_trap[1:-1] = V[1:-1]*phi[1:-1] # potential term

    k1[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*phi[1:-1])
    
    # Neumann Boundary Conditions
    k1[0] = k1[1]
    k1[-1] = k1[-2]
    
    # k2 CALCULATION
    KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k1) + Dr2 @ k1) # Kinetic Energy derivatives
    # HAMILTONIAN TERMS
    H_KE[1:-1] = -0.5*KE[1:-1] # KE term
    H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k1[1:-1]/2)**3*(phi[1:-1] + k1[1:-1]/2) # LHY term
    H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k1[1:-1]/2)**2*(phi[1:-1] + k1[1:-1]/2) # s-wave term
    H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k1[1:-1]/2) # potential term
    
    k2[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k1[1:-1]/2))
    
    # Neumann Boundary Conditions
    k2[0] = k2[1]
    k2[-1] = k2[-2]
    
    # k3 CALCULATION
    KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k2) + Dr2 @ k2) # Kinetic Energy derivatives 
    # HAMILTONIAN TERMS 
    H_KE[1:-1] = -0.5*KE[1:-1] # KE term
    H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k2[1:-1]/2)**3*(phi[1:-1] + k2[1:-1]/2) # LHY term
    H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k2[1:-1]/2)**2*(phi[1:-1] + k2[1:-1]/2) # s-wave term
    H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k2[1:-1]/2) # potential term
   
    k3[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k2[1:-1]/2))
    
    # Neumann Boundary Conditions
    k3[0] = k3[1]
    k3[-1] = k3[-2]
    
    # k4 CALCULATION
    KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + ((2/r[1:-1])*(Dr @ k3) + Dr2 @ k3) # Kinetic Energy derivatives 
    # HAMILTONIAN TERMS
    H_KE[1:-1] = -0.5*KE[1:-1] # KE term
    H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k3[1:-1])**3*(phi[1:-1] + k3[1:-1]) # LHY term
    H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k3[1:-1])**2*(phi[1:-1] + k3[1:-1]) # s-wave term
    H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k3[1:-1]) # potential term
    
    k4[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k3[1:-1]))
    
    # Neumann Boundary Conditions
    k4[0] = k4[1]
    k4[-1] = k4[-2]
    
    # FINAL RUNGE-KUTTA STEP    
    phi[1:-1] = phi[1:-1] + (1./6)*(k1[1:-1] + 2*k2[1:-1] + 2*k3[1:-1] + k4[1:-1])
    
    # NEUMANN BOUNDARY CONDITIONS
    # phi(j+1) - phi(j) = 0
    phi[0] = phi[1]
    phi[-1] = phi[-2]
    
    # IN TIME PLOTTING (focused upon the boundaries of the box)
    if (l % t_frame == 0):
        f, axs = plt.subplots(1,2,figsize=(12,8))
        plt.subplot(1,2,1)
        plt.plot(r,N*np.abs(phi)**2)
        plt.xlim((0,10*dr))
        plt.ylim((1.098,1.1))
        plt.xlabel("$r$")
        plt.ylabel("$n$")
        plt.title("$N = $"+str(N)+" t = "+str(round(t)))
        plt.subplot(1,2,2)
        plt.plot(r,N*np.abs(phi)**2)
        plt.xlim((Lr-10*dr,Lr))
        plt.ylim((0,0.001))
        plt.xlabel("$r$")
        plt.ylabel("$n$")
        plt.title("$N = $"+str(N)+" t = "+str(round(t)))       
        plt.pause(0.2)
        print('l = ',l,'of ',t_steps,'steps (',100*(l/t_steps),'%)')
        
    # SAVING DATA AND OBSERVABLES
    if (l % t_save == 0):
        spacetime[:,l//t_save] = phi # save current wavefunction
        phase[:,l//t_save] = np.angle(phi) # save current phase of wavefunction
        t_array[l//t_save] = t # save current time
        mean_r2[l//t_save] = 4*pi*np.trapz(r**4*np.abs(phi)**2)*dr # save current observable <r^2>
        
    # ITERATE TIME     
    t = t + dt   
    
    # ITERATE COUNTER
    count = count + 1

# PLOTTING <r^2(t)>
plt.plot(t_array,mean_r2)
plt.xlim((0,max(t_array)))
plt.xlabel("$t$")
plt.ylabel("$<r^2>$")
#plt.title("$\lambda = $"+str(lamb)+", $\mu_{tol} = $"+str(tol)+", $N = $"+str(N))
plt.savefig("quench_3000.png",dpi=300)
#plt.clf() # clear plot

# PLOTTING FINAL DENSITY SNAPSHOT
#plt.plot(r,abs(phi)**2)
#plt.xlim((0,Lr))
#plt.ylim((0,2))
#plt.xlabel("$r$")
#plt.ylabel("$n(r)$")
#plt.savefig("dens_m_par.png",dpi=300)
"""
# CURVE FITTING
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

# Extract the b coefficient as this is the frequency of the sine curve
omega = popt[1]
print("omega = ",omega)

# Create the fitted curve
fitted = sine_func(t_array,popt[0],popt[1],popt[2],popt[3],pop[4])

# Plot the fitted curve on top of the data
#plt.plot(t_array,fitted,':')
"""
