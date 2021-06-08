# %%
"""
# Mapping from experimental parameters to Petrov effective atom number
"""

# %%
"""
## Load in packages, data and function
"""

# %%
"""
### Packages
"""

# %%
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

# %%
"""
### Physical constants
"""

# %%
pi = np.pi
hbar = 1.0545718*1e-34
a0 = 5.29*1e-11 # Bohr radius
Da = 1.66053906660*1e-27

# %%
"""
### Parameters
"""

# %%
m1 = 133*Da # mass 1 (Cs)
m2 = 174*Da # mass 2 (Yb-174)
a12 = -75*a0 # interspecies scattering length
a22 = 105*a0 # intraspecies 2 scattering length

# %%
"""
### Function to generate either Petrov parameter or critical atom number
"""

# %%
def pet_N(m1,m2,a11,a22,a12,N,crit):
    """
    This function can be used in two ways:
    
    1) If crit == 0 then the function will calculate for given: scatterings lengths, atomic masses and
       particle number, the associated Petrov effective particle number \tilde{N}. This can then be inputted
       into an imaginary time loop to find the groundstate and to check for the droplet profile.
       
    !OR!
       
    2) If crit == 1 then the function will calculate for given: scattering lengths and atomic masses, the
       critical total particle number needed for a self-bound groundstate. This does however mean that no
       particle number input is needed so any value can be inputted as it is not used in the calculation.
       This computation is essentially a ratio between the experimental parameters and the critical
       value of the Petrov effective particle number \tilde{N}_c = 18.65.
       
    To use crit == 0 it can be useful to embed this into a double for loop to iterate across (N,\deltag)
    phase space. However, it can simply be used as a mapping tool to map from all known experimental
    parameters to the Petrov effective particle number. On the other hand crit == 1 can be used to create
    a plot, by embedding in a single for loop, of a free scattering length against the particle number and
    plotting the critical N for the self-bound state.
    """
    # LHY TERM INTEGRAND (in k)
    def F(k,z,x):
        pet_eq =  (k**2*np.sqrt(0.5*(k**2*(1+x/z) + 0.25*k**4*(1+z**(-2))) + np.sqrt(0.25*((k**2 + 0.25*k**4)-((x/z)*k**2 + 0.25*z**(-2)*k**4))**2 + (x/z)*k**4)) 
        + k**2*np.sqrt(0.5*(k**2*(1+x/z) + 0.25*k**4*(1+z**(-2))) - np.sqrt(0.25*((k**2 + 0.25*k**4)-((x/z)*k**2 + 0.25*z**(-2)*k**4))**2 + (x/z)*k**4))
        -((1+z)/(2*z))*k**4 - (1+x)*k**2 + (1/(1+z))*((1+x*z)**2 + z*(1+x)**2))
        return pet_eq

    # LHY TERM INTEGRAND (in t)
    def Ft(t,z,x):
        pet_eq = np.cos(t)**(-2)*(np.tan(t)**2*np.sqrt(0.5*(np.tan(t)**2*(1+x/z) + 0.25*np.tan(t)**4*(1+z**(-2))) + np.sqrt(0.25*((np.tan(t)**2 + 0.25*np.tan(t)**4)-((x/z)*np.tan(t)**2 + 0.25*z**(-2)*np.tan(t)**4))**2 + (x/z)*np.tan(t)**4)) 
    + np.tan(t)**2*np.sqrt(0.5*(np.tan(t)**2*(1+x/z) + 0.25*np.tan(t)**4*(1+z**(-2))) - np.sqrt(0.25*((np.tan(t)**2 + 0.25*np.tan(t)**4)-((x/z)*np.tan(t)**2 + 0.25*z**(-2)*np.tan(t)**4))**2 + (x/z)*np.tan(t)**4))
    -((1+z)/(2*z))*np.tan(t)**4 - (1+x)*np.tan(t)**2 + (1/(1+z))*((1+x*z)**2 + z*(1+x)**2))
        return pet_eq
    
    # PHYSICAL CONSTANTS
    pi = np.pi
    hbar = 1.0545718*1e-34
    a0 = 5.29*1e-11 # Bohr radius
    Da = 1.66053906660*1e-27
    
    # INTERACTION STRENGTHS
    g11 = 4*pi*hbar**2*a11/m1
    g12 = 2*pi*hbar**2*a12*(1/m1 + 1/m2)
    g22 = 4*pi*hbar**2*a22/m2
    deltag = g12 + np.sqrt(g11*g22)
    
    # CALCULATING EQUILIBRIUM DENSITIES
    if m1 == m2:
        # Equilibrium densities (equal masses)
        n01 = (25*pi/1024)*(a12+np.sqrt(a11*a22))**2/(a11**1.5*a22*(np.sqrt(a11) + np.sqrt(a22))**5)
        n02 = (25*pi/1024)*(a12+np.sqrt(a11*a22))**2/(a22**1.5*a11*(np.sqrt(a11) + np.sqrt(a22))**5)
    else:
        # Compute the integral f using the tan(t) transformation and integrate across [0,pi/2]
        f = (15/32)*integrate.quadrature(lambda t,z,x: Ft(t,z,x),0,pi/2,args=(m2/m1,np.sqrt(g22/g11)),rtol=1e-06,maxiter=5000)[0]
        # Equilibrium densities (unequal masses)
        n01 = (25*pi/1024)*(deltag**2/(f**2*a11**3*g11*g22))
        n02 = n01*np.sqrt(g11/g22)
    
    # LENGTH AND TIMESCALES
    xi = hbar*np.sqrt(1.5*(np.sqrt(g22)/m1 + np.sqrt(g11)/m2)/(np.abs(deltag)*np.sqrt(g11)*n01))
    tau = hbar*1.5*(np.sqrt(g11) + np.sqrt(g22))/(np.abs(deltag)*np.sqrt(g11)*n01)
    
    # FINAL PARTICLE NUMBER CALCULATION
    if crit == 0:
        # PETROV PARAMETER
        N_tilde = (N/(n01*xi**3))*(np.sqrt(g22)/(np.sqrt(g11) + np.sqrt(g22)))
    elif crit == 1:
        # CRITICAL TOTAL PARTICLE NUMBER FOR DROPLET
        N_tilde = (18.65*n01*xi**3)*((np.sqrt(g11) + np.sqrt(g22))/np.sqrt(g22))
        
    return N_tilde,xi,tau,n01

# %%
"""
## Script to call parameter mapping function
"""

# %%
"""
### Choose either critical N finder (crit == 1) or Petrov \tilde{N} parameter
"""

# %%
crit = 0

# %%
"""
### Script to call Petrov parameter mapping function for: 
#### -> (crit == 0) generating Petrov parameter for user inputted values of a11 and N ####
#### -> (crit == 1) the critical atom number needed for user inputted value of a11####
"""

# %%
if crit == 0:
    # user inputs
    a11 = float(input("Input value of $a_{11}$"))*a0
    N = float(input("Now input the total atom number, N"))
    # Find Petrov parameter N for given parameters
    N_tilde,xi,tau,n01 = pet_N(m1,m2,a11*a0,a22,a12,N,crit)
    print("Petrov parameter N is = ",N_tilde)
elif crit == 1:
    # user input
    a11 = float(input("Input value of $a_{11}$"))*a0
    # Find the critical atom number for given parameter
    N_tilde,xi,tau,n01 = pet_N(m1,m2,a11,a22,a12,N,crit)
    print("The critical atom number $N_c$ is",N_tilde)

# %%
"""
## Experimentally necessary figures
"""

# %%
"""
### (N,a11) Phase plot of droplet width and peak density
"""

# %%
"""
This code will construct two phase diagrams across the (N,a11) phase space, one of the average droplet width <r^2> and one of the maximum droplet density max(phi). To plot these phase diagrams, matplotlib plots can be saved and/or data can be saved to be inputted to pgfplots. For further details on how the data is saved for pgfplots, please see the commented code below.

**WARNING:** Make sure to check your local directory to ensure there are no 'width_a11.dat' or 'mode_a11.dat' files as the pgfplots data saving script will not overwrite and will simply add to this file which will not be read in my pgfplots.
"""

# %%
# LOAD RUNGE-KUTTA IMAGINARY TIME LOOP
from petrov_im_tim_rk4 import petrov_im_tim_rk4_mat # RK4 imaginary time function

# PARAMETERS
pi = np.math.pi

# FREE SCATTERING LENGTH ARRAY SETUP
a11_max = 40
a11_min = 10
a11_step = 1
a11_ARRAY = np.arange(a11_min,a11_max+1,a11_step)

# TOTAL ATOM NUMBER ARRAY SETUP
N_min = 54000
N_max = 75000
N_step = 5*1e3
N_ARRAY = np.arange(N_min,N_max+1,N_step)

# SPECIFY ARRAYS OF ZEROS FOR DATA TO BE SAVED: Ntilde, droplet width and max density (mode)
Ntilde_MAT = np.zeros((len(N_ARRAY),len(a11_ARRAY)))
width_MAT = np.zeros((len(N_ARRAY),len(a11_ARRAY)))
mode_MAT = np.zeros((len(N_ARRAY),len(a11_ARRAY)))

# GRID
Lr = 32 # box length
Nr = 256 # grid points
dr = Lr/Nr # spatial step
r = np.arange(-1/2,(Nr + 3/2),1)*dr # position array with 4 ghost points

# TIME SETUP
dt = 0.1*dr**2 # time step 
im_t_steps = 250000 # number of imaginary time steps

# POTENTIAL
V = np.zeros(r.size)

# INITIALISE WAVEFUNCTION
phi_0 = np.exp(-(r)**2/(2*(5)**2)) # Gaussian initial condition (do not set width = 1 or width too large)
Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm) # Normalised initial condition

# ITERATE ACROSS THE a11 ARRAY AND CALCULATE Ntilde FOR EACH VALUE OF a11
for i in range(0,len(N_ARRAY)):
    for j in range(0,len(a11_ARRAY)):
        # Calculate the Petrov parameter for specific a11 and N values
        Ntilde,xi,tau,n01 = pet_N(m1,m2,a11_ARRAY[j]*a0,a22,a12,N_ARRAY[i],crit=0)
        # Save Petrov parameter into 2D array
        Ntilde_MAT[i,j] = Ntilde
        # Set up effective interaction strength and the density scaling
        g11 = 4*pi*hbar**2*a11_ARRAY[j]*a0/m1
        dens_scale = n01*(np.sqrt(g11) + np.sqrt(g22))/np.sqrt(g22)
        
        # IMAGINARY TIME
        [phi,mu,tol_mode] = petrov_im_tim_rk4_mat(phi_0,r,dr,dt,Ntilde,V,1,im_t_steps)
        # Plot groundstate solution
        print("N = ",Ntilde)
        plt.plot(r,Ntilde*np.abs(phi)**2)
        plt.show()
        # Save dimensional average width of the droplet
        width_MAT[i,j] = Ntilde*(dens_scale*xi**5)*(4*pi*np.trapz(r**4*np.abs(phi)**2)*dr)
        # Save dimensional peak density of the droplet
        mode_MAT[i,j] = dens_scale*Ntilde*max(np.abs(phi)**2)
        
# Meshgrid the (N,a11) phase space
A,N = np.meshgrid(a11_ARRAY,N_ARRAY)

# MATPLOTLIB PCOLOR PLOTS
#plt.pcolor(A,N,width_MAT,shading="auto")
#plt.xlabel("a11")
#plt.ylabel("N")
#clb1 = plt.colorbar()
#clb1.ax.set_title('xi')
#plt.savefig('widtha11.png',dpi=300)

#clb1.remove()
#plt.pcolor(A,N,mode_MAT,shading="auto")
#plt.colorbar()
#plt.xlabel("a11")
#plt.ylabel("N")
#clb2 = plt.colorbar()
#clb2.ax.set_title('max[n_0(r)]')
#plt.savefig('dimful_mode_quad.png',dpi=300)

# SAVING DATA
N = N.flatten() # Flatten the N array into a 1D array
A = A.flatten() # Flatten the a11 array into a 1D array
width_MAT = width_MAT.flatten() # Flatten the <r^2> array into a 1D array
mode_MAT = mode_MAT.flatten() # Flatten the max(phi) array into a 1D array
wa11_data = np.column_stack((N,A,width_MAT)) # Stack the 1D arrays to form 3 columns for the <r^2> data
ma11_data = np.column_stack((N,A,mode_MAT)) # Stack the 1D arrays to form 3 columns for the max(phi) data

# FORMATTING DATA FOR PGFPLOTS TO READ INTO A COLOURMAP PLOT
for j in range(0,wa11_data.shape[0]): # Iterating across both 3 column <r^2> and max(phi) data
    # This is just to make sure the final line is printed also as this was frequently a problem
    if j == wa11_data.shape[0]-1:
        # width data
        with open ('width_a11.dat','a') as fw:
            fw.write(str(wa11_data[j,1]) + ' ' + str(wa11_data[j,0]) + ' ' +str(wa11_data[j,2])+'\n')
        # mode data
        with open ('mode_a11.dat','a') as fm:
            fm.write(str(ma11_data[j,1]) + ' ' + str(ma11_data[j,0]) + ' ' +str(ma11_data[j,2])+'\n')
    # If the value in the first column changes, save the data and add another line for formatting
    # this is because pgfplots needs a gap in the data to register a new line for the plot is needed
    elif wa11_data[j+1,0] != wa11_data[j,0]:
        # width data
        with open ('width_a11.dat','a') as fw:
            fw.write(str(wa11_data[j,1]) + ' ' + str(wa11_data[j,0]) + ' ' +str(wa11_data[j,2]) + '\n'+'\n')
        # mode data
        with open ('mode_a11.dat','a') as fm:
            fm.write(str(ma11_data[j,1]) + ' ' + str(ma11_data[j,0]) + ' ' +str(ma11_data[j,2]) + '\n'+'\n')
    # Finally, if the first column does not change and the data is not the final line then this part of
    # of the if statement tree simply just saves the data line and starts on the next line
    else:
        # width data
        with open ('width_a11.dat','a') as fw:
            fw.write(str(wa11_data[j,1]) + ' ' + str(wa11_data[j,0]) + ' ' +str(wa11_data[j,2]) + '\n')
        # mode data
        with open ('mode_a11.dat','a') as fm:
            fm.write(str(ma11_data[j,1]) + ' ' + str(ma11_data[j,0]) + ' ' +str(ma11_data[j,2]) + '\n')
# Close the data files
fw.close()
fm.close()

# %%
"""
### Parameters (xi,tau) as a function of a11
"""

# %%
crit = 1

# setting up input arrays
N = 1
a11_max = 40
a11_min = 10
a11_step = 0.5
a11_ARRAY = np.arange(a11_min,a11_max+1,a11_step)
xi_ARRAY = np.zeros(len(a11_ARRAY))
tau_ARRAY = np.zeros(len(a11_ARRAY))
Ncrit_ARRAY = np.zeros(len(a11_ARRAY))

# ITERATE ACROSS THE a11 ARRAY AND CALCULATE N_{crit} FOR EACH VALUE OF a11
for i in range(0,len(a11_ARRAY)):
    Ncrit,xi,tau,n01 = pet_N(m1,m2,a11_ARRAY[i]*a0,a22,a12,N,crit)
    Ncrit_ARRAY[i] = Ncrit
    xi_ARRAY[i] = xi
    tau_ARRAY[i] = tau
    
# PRINTING 
print("a_11 = ",a11_ARRAY)
print("N_c = ",Ncrit_ARRAY)
g11_ARRAY = 4*pi*hbar**2*(a11_ARRAY*a0)/m1
deltag_ARRAY = g12 + np.sqrt(g11_ARRAY*g22)
print("deltag = ",deltag_ARRAY)

# PLOTTING 
#fig = plt.figure()
#plt.figure(figsize=(15,15))

# Critical Atom Number Vs a11
#plt.subplot(2, 2, 1)
#plt.plot(a11_ARRAY,Ncrit_ARRAY)
#plt.xlabel(r'$a_{11}[a_{0}]$')
#plt.ylabel(r'$N_{c}$')
#plt.savefig('a11Nc.png',dpi=300)
#plt.clf()

# xi Vs a11
#plt.subplot(2, 2, 2)
#plt.plot(a11_ARRAY,xi_ARRAY*1e6)
#plt.xlabel(r'$a_{11}[a_{0}]$')
#plt.ylabel(r'$\xi[\mu m]$')
#plt.savefig('a11xi.png',dpi=300)
#plt.clf()

# tau Vs a11
#plt.subplot(2, 2, 3)
#plt.plot(a11_ARRAY,tau_ARRAY*1e3)
#plt.xlabel(r'$a_{11}[a_{0}]$')
#plt.ylabel(r'$\tau \, [ms]$')
#plt.savefig('a11tau.png',dpi=300)
#plt.show()

# DATA SAVING
Nc_data = np.column_stack((a11_ARRAY,Ncrit_ARRAY)) # Critical atom number
np.savetxt('Ncdat_improved.csv',Nc_data,delimiter=',',fmt='%18.16f')

xi_data = np.column_stack((a11_ARRAY,xi_ARRAY*1e6)) # Length scale
np.savetxt('xidat_improved.csv',xi_data,delimiter=',',fmt='%18.16f')

tau_data = np.column_stack((a11_ARRAY,tau_ARRAY*1e3)) # Time scale
np.savetxt('taudat_improved.csv',xi_data,delimiter=',',fmt='%18.16f')