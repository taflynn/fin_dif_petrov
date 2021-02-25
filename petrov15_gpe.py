# PACKAGES
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

# SWITCHES:
# Method (Euler = 0, RK4 = 1)
method = 1
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
t_steps = 500000 # number of real time steps
im_t_steps = 500000 # number of imaginary time steps
t_frame = 10000
t_save = 100

# PARAMETERS
N = 1314.65 # effective atom number
pi = np.math.pi

# POTENTIAL
if trap == 0:
	V = np.zeros(r.size)
elif trap == 1:
	V = 0.5*r**2

# INITIALISE WAVEFUNCTION
phi_0 = np.exp(-(r)**2/(2*(10)**2)) # Gaussian initial condition
Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm) # normalised initial condition
phi = phi_0

# GPE COEFFICIENTS
if int_gas == 0:
	int_coef = 0
	LHY_coef = 0
elif int_gas == 1:    
	int_coef = -3*N
	LHY_coef = (5/2)*N**(3/2)    

# INITIAL MU
phi_r = np.gradient(phi,dr) # derivative of wavefunction
mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2  + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
tol = 1 # initialise tolerance
count = 1 # initialise imaginary time counter

# DIFFERENTIAL OPERATORS IN MATRIX FORM (Dr - first order, Dr2 - second order)
Dr = (1/(2*dr))*(-1*eye(phi.size-2,phi.size,k=0,dtype=float) + eye(phi.size-2,phi.size,k=2,dtype=float))
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
 
t_array = np.zeros((t_steps//t_save)) 
mean_r2 = np.zeros((t_steps//t_save)) # observable used here <r^2> 
dens_current = max(phi)
###############################################################################
# IMAGINARY TIME LOOP:
    
for l in range(0,im_t_steps):
	# k1 CALCULATION
	KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi
	# HAMILTONIAN TERMS
	H_KE[1:-1] = -0.5*KE[1:-1] # KE term
	H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1])**3*phi[1:-1] # LHY term
	H_int[1:-1] = int_coef*np.abs(phi[1:-1])**2*phi[1:-1] # s-wave term
	H_trap[1:-1] = V[1:-1]*phi[1:-1] # potential term

	k1[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1])

	# k2 CALCULATION
	KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k1) + Dr2 @ k1) 
	# HAMILTONIAN TERMS
	H_KE[1:-1] = -0.5*KE[1:-1] # KE term
	H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k1[1:-1]/2)**3*(phi[1:-1] + k1[1:-1]/2) # LHY term
	H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k1[1:-1]/2)**2*(phi[1:-1] + k1[1:-1]/2) # s-wave term
	H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k1[1:-1]/2) # potential term
    
	k2[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1])
    
	# k3 CALCULATION
	KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k2) + Dr2 @ k2)  
	# HAMILTONIAN TERMS 
	H_KE[1:-1] = -0.5*KE[1:-1] # KE term
	H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k2[1:-1]/2)**3*(phi[1:-1] + k2[1:-1]/2) # LHY term
	H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k2[1:-1]/2)**2*(phi[1:-1] + k2[1:-1]/2) # s-wave term
	H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k2[1:-1]/2) # potential term
   
	k3[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1])

	#k4 CALCULATION
	KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + ((2/r[1:-1])*(Dr @ k3) + Dr2 @ k3)  
	# HAMILTONIAN TERMS
	H_KE[1:-1] = -0.5*KE[1:-1] # KE term
	H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k3[1:-1])**3*(phi[1:-1] + k3[1:-1]) # LHY term
	H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k3[1:-1])**2*(phi[1:-1] + k3[1:-1]) # s-wave term
	H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k3[1:-1]) # potential term
    
	k4[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1])
    
	# FINAL RUNGE-KUTTA STEP    
	phi[1:-1] = phi[1:-1] + (1./6)*(k1[1:-1] + 2*k2[1:-1] + 2*k3[1:-1] + k4[1:-1])
   
	# NEUMANN BOUNDARY CONDITIONS
	# phi(j+1) - phi(j) = 0
	k1[0] = k1[1]
	k1[-1] = k1[-2]
	k2[0] = k2[1]
	k2[-1] = k2[-2]
	k3[0] = k3[1]
	k3[-1] = k3[-2]
	k4[0] = k4[1]
	k4[-1] = k4[-2]
	phi[0] = phi[1]
	phi[-1] = phi[-2]

	# WAVEFUNCTION NORMALISED
	Norm = 4*pi*np.trapz(r**2*abs(phi)**2)*dr
	phi = phi/np.sqrt(Norm) 
    
	# MU AND TOLERANCE CALCULATION
	mu_old = mu
	phi_r = np.gradient(phi,dr)
	mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2 + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
	tol = abs((mu-mu_old)/mu_old)
	# TOLERANCE BETWEEN SUCCESSIVE MAX DENSITIES
	dens_prev = dens_current
	dens_current = max(phi)
	tol_dens = abs((dens_current - dens_prev)/dens_prev)
    
	# PLOTTING AND SOME IMAGINARY TIME OUTPUT
	if (l % 10000 == 0):
        #plt.plot(r,N*abs(phi)**2)
        #plt.xlim((0,Lr))
        #plt.ylim((0,2))
        #plt.xlabel("$r$")
        #plt.ylabel("$n$")
        #plt.title("$N = $"+str(N))
        #plt.pause(0.2)
		print('l = ',l)
    
	# ITERATE COUNTER
	count = count + 1

###############################################################################
# REAL TIME LOOP:

phi_ground = phi

print("Groundstate found with $\mu_{tol}$",tol,"dens_tol",tol_dens)

# plotting the groundstate at the extrema of the box
f, axs = plt.subplots(1,2,figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(r,N*abs(phi)**2)
plt.xlim((0,10*dr))
plt.ylim((1.00,1.15))
plt.xlabel("$r$")
plt.ylabel("$n$")
plt.title("$N = $"+str(N))
plt.subplot(1,2,2)
plt.plot(r,N*abs(phi)**2)
plt.xlim((Lr-10*dr,Lr))
plt.ylim((0,0.001))
plt.xlabel("$r$")
plt.ylabel("$n$")
plt.title("$N = $"+str(N))       
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
phi = phi.astype(complex)
t = 0
spacetime = np.zeros((r.size,(t_steps//t_save))).astype(complex)
phase = np.zeros((r.size,(t_steps//t_save))).astype(complex)

# swap to a smaller time step in real time
dt = 0.1*dr**2

print('now multiplying by the small phase')

# invoke breathing mode
lamb = 1e-4
phi = np.exp(1j*lamb*r**2)*phi

# plotting density after the small phase imprint
f, axs = plt.subplots(1,2,figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(r,N*abs(phi)**2)
plt.xlim((0,10*dr))
plt.ylim((1.00,1.15))
plt.xlabel("$r$")
plt.ylabel("$n$")
plt.title("$N = $"+str(N))
plt.subplot(1,2,2)
plt.plot(r,N*abs(phi)**2)
plt.xlim((Lr-10*dr,Lr))
plt.ylim((0,0.001))
plt.xlabel("$r$")
plt.ylabel("$n$")
plt.title("$N = $"+str(N))       
plt.pause(0.2)

print('real time starting now:')

for l in range(0,t_steps):  
	# k1 CALCULATION
	KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi
	# HAMILTONIAN TERMS
	H_KE[1:-1] = -0.5*KE[1:-1] # KE term
	H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1])**3*phi[1:-1] # LHY term
	H_int[1:-1] = int_coef*np.abs(phi[1:-1])**2*phi[1:-1] # s-wave term
	H_trap[1:-1] = V[1:-1]*phi[1:-1] # potential term

	k1[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1])
    
	# k2 CALCULATION
	KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k1) + Dr2 @ k1) 
	# HAMILTONIAN TERMS
	H_KE[1:-1] = -0.5*KE[1:-1] # KE term
	H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k1[1:-1]/2)**3*(phi[1:-1] + k1[1:-1]/2) # LHY term
	H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k1[1:-1]/2)**2*(phi[1:-1] + k1[1:-1]/2) # s-wave term
	H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k1[1:-1]/2) # potential term
    
	k2[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1])
    
	# k3 CALCULATION
	KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k2) + Dr2 @ k2)  
	# HAMILTONIAN TERMS 
	H_KE[1:-1] = -0.5*KE[1:-1] # KE term
	H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k2[1:-1]/2)**3*(phi[1:-1] + k2[1:-1]/2) # LHY term
	H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k2[1:-1]/2)**2*(phi[1:-1] + k2[1:-1]/2) # s-wave term
	H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k2[1:-1]/2) # potential term
   
	k3[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1])
    
	# k4 CALCULATION
	KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + ((2/r[1:-1])*(Dr @ k3) + Dr2 @ k3)  
	# HAMILTONIAN TERMS
	H_KE[1:-1] = -0.5*KE[1:-1] # KE term
	H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k3[1:-1])**3*(phi[1:-1] + k3[1:-1]) # LHY term
	H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k3[1:-1])**2*(phi[1:-1] + k3[1:-1]) # s-wave term
	H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k3[1:-1]) # potential term
    
	k4[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1])
    
	# FINAL RUNGE-KUTTA STEP    
	phi[1:-1] = phi[1:-1] + (1./6)*(k1[1:-1] + 2*k2[1:-1] + 2*k3[1:-1] + k4[1:-1])
    
	# NEUMANN BOUNDARY CONDITIONS
	# phi(j+1) - phi(j) = 0
	k1[0] = k1[1]
	k1[-1] = k1[-2]
	k2[0] = k2[1]
	k2[-1] = k2[-2]
	k3[0] = k3[1]
	k3[-1] = k3[-2]
	k4[0] = k4[1]
	k4[-1] = k4[-2]
	phi[0] = phi[1]
	phi[-1] = phi[-2]
    
	# PLOTTING
	if (l % t_frame == 0):
		f, axs = plt.subplots(1,2,figsize=(12,8))
		plt.subplot(1,2,1)
		plt.plot(r,N*abs(phi)**2)
		plt.xlim((0,10*dr))
		plt.ylim((1.00,1.15))
		plt.xlabel("$r$")
		plt.ylabel("$n$")
		plt.title("$N = $"+str(N)+" t = "+str(round(t)))
		plt.subplot(1,2,2)
		plt.plot(r,N*abs(phi)**2)
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
		phase[:,l//t_save] = np.angle(phi)
		t_array[l//t_save] = t
		mean_r2[l//t_save] = 4*pi*np.trapz(r**4*abs(phi)**2)*dr
       
	# ITERATE TIME     
	t = t + dt   
    
	# ITERATE COUNTER
	count = count + 1

plt.plot(t_array,mean_r2)
plt.xlim((0,max(t_array)))
plt.xlabel("$t$")
plt.ylabel("$<r^2>$")
plt.title("$\lambda = $"+str(lamb)+", $\mu_{tol} = $"+str(tol)+", $N = $"+str(N))
