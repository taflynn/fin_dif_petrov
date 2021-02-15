# PACKAGES
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

# start timer
tic = time.perf_counter()

# SWITCHES:
# Method (Euler = 0, RK4 = 1)
method = 1
# trap? 0=NO,1=YES
trap = 0
# interacting gas? (0 = Non-interacting gas, 1 = Interacting gas w/ LHY)
int_gas = 1

# GRID
Lr = 12 # box length
Nr = 128 # grid points
dr = Lr/Nr # spatial step
r = np.arange(-1/2,(Nr + 3/2),1)*dr # position array with 4 ghost points

# TIME SETUP
dt = 0.1*dr**2 # time step 
t_steps = 1000000 # number of real time steps
im_t_steps = 1000000 # number of imaginary time steps

# PARAMETERS
N = 19 # effective atom number
pi = np.math.pi

# POTENTIAL
if trap == 0:
    V = 0
elif trap == 1:
    V = 0.5*r**2

# INITIALISE WAVEFUNCTION
phi_0 = np.exp(-(r)**2/(2*(1)**2)) # Gaussian initial condition
phi_0[0] = phi_0[1]
phi_0[-1] = phi_0[-2]
#phi_0 = np.ones(len(r))
Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm) # normalised initial condition
phi = phi_0
# IMAGINARY TIME
#if method == 0:
    #[phi, mu, count] = petrov_im_tim_euler(phi_0,r,dr,dt,N,V)
#elif method == 1:
    #[phi,mu,tol_mu,tol_mode] = petrov_im_tim_rk4_mat(phi_0,r,dr,dt,N,V,int_gas,im_t_steps)
    #[phi,mu] = petrov_im_tim_rk4(phi_0,r,dr,dt,N,V,int_gas)
    
# end timer    
toc = time.perf_counter()
print(toc-tic)

# PLOTTING GROUNDSTATE WAVEFUNCTION
#plt.plot(r,np.sqrt(N)*phi)
#plt.xlim((0,Lr))
#plt.ylim((0,1.2))
#plt.xlabel("$r$")
#plt.ylabel("$\phi_0$")
#plt.title("$N = $"+str(N))
#plt.show()
# PRINTING OUT TOLERANCES FROM IMAGINARY TIME FUNCRION
#print("tol_mu = ",tol_mu)
#print("tol_mode = ",tol_mode)
# SAVING DATA (2 columns, spatial array and wavefunction)
#DataOut = np.column_stack((r,np.sqrt(N)*phi))
#np.savetxt('phi0_N'+str(N)+'prime.csv',DataOut,delimiter=',',fmt='%18.16f')

###################################################################################################

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

# DIFFERENTIAL OPERATORS
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
# 1st-order derivative
#d11 = np.zeros(r.size-1); d12 = -1*np.ones(r.size); d13 = np.ones(r.size-1)
d11 = -1*np.ones(r.size-1); d12 = -1*np.zeros(r.size); d13 = np.ones(r.size-1); 
#d11 = -1*np.ones(r.size-1); d12 = np.ones(r.size); d13 = np.zeros(r.size-1)

Dr = (1/dr)*tridiag(d11,d12,d13)
Dr[0,-1] = 1/dr; Dr[-1,0] = -1/dr;
# 2nd-order derivative
d21 = np.ones(r.size-1); d22 = -2*np.ones(r.size); d23 = np.ones(r.size-1)
Dr2 = (1/dr**2)*tridiag(d21, d22, d23)

# INITIALISING ARRAYS
H_KE = np.zeros(phi.size)
H_LHY = np.zeros(phi.size)
H_int = np.zeros(phi.size)
H_trap = np.zeros(phi.size)
KE = np.zeros(phi.size)

t = 0 # initial time 
t_array = np.zeros(im_t_steps+1) 
mean_r2 = np.zeros(im_t_steps+1) # observable used here <r^2> 
phi_dr2 = np.zeros(im_t_steps+1)
phi_3dr2 = np.zeros(im_t_steps+1)
'''
for l in range(0,im_t_steps+1):
    # k1 CALCULATION
    for j in range(1,phi.size-1):
        KE[j] = (2/r[j])*(phi[j+1] - phi[j-1])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2  
    # HAMILTONIAN TERMS    
    H_KE = -0.5*KE # KE term
    H_LHY = LHY_coef*abs(phi)**3*phi # LHY term
    H_int = int_coef*abs(phi)**2*phi # s-wave term
    H_trap = V*phi # potential term
    
    k1 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*phi)
    
    # k2 CALCULATION
    for j in range(1,phi.size-1):
        KE[j] = (2/r[j])*(phi[j+1] - phi[j-1])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2 + 0.5*(2/r[j])*(k1[j+1] - k1[j-1])/dr + 0.5*(k1[j+1] - 2*k1[j] + k1[j-1])/dr**2  
    # HAMILTONIAN TERMS
    H_KE = -0.5*KE # KE term
    H_LHY = LHY_coef*abs(phi + k1/2)**3*(phi + k1/2) # LHY term
    H_int = int_coef*abs(phi + k1/2)**2*(phi + k1/2) # s-wave term
    H_trap = V*(phi + k1/2) # potential term
    
    k2 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*(phi + k1/2))
   
    # k3 CALCULATION
    for j in range(1,phi.size-1):
        KE[j] = (2/r[j])*(phi[j+1] - phi[j-1])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2 + 0.5*(2/r[j])*(k2[j+1] -k2[j-1])/dr + 0.5*(k2[j+1] - 2*k2[j] + k2[j-1])/dr**2  
    # HAMILTONIAN TERMS
    H_KE = -0.5*KE
    H_LHY = LHY_coef*abs(phi + k2/2)**3*(phi + k2/2)
    H_int = int_coef*abs(phi + k2/2)**2*(phi + k2/2)
    H_trap = V*(phi + k2/2)
    
    k3 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*(phi + k2/2))

    #k4 CALCULATION
    for j in range(1,phi.size-1):
        KE[j] = (2/r[j])*(phi[j+1] - phi[j-1])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2 + (2/r[j])*(k3[j+1] -k3[j-1])/dr + (k3[j+1] - 2*k3[j] + k3[j-1])/dr**2  
    # HAMILTONIAN TERMS
    H_KE = -0.5*KE # KE term
    H_LHY = LHY_coef*abs(phi + k3)**3*(phi + k3) # LHY term
    H_int = int_coef*abs(phi + k3)**2*(phi + k3) # s-wave term
    H_trap = V*(phi + k3) # potential term
    
    k4 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*(phi + k3))

    # FINAL RUNGE-KUTTA STEP    
    phi = phi + (1./6)*(k1 + 2*k2 + 2*k3 + k4)
   
    # NEUMANN BOUNDARY CONDITIONS
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
    if (l % 5000 == 0):
        plt.plot(r,np.sqrt(N)*phi)
        plt.xlim((0,Lr))
        plt.ylim((0,2))
        plt.xlabel("$r$")
        plt.ylabel("$\phi_0$")
        plt.title("$N = $"+str(N))
        plt.pause(0.2)
        print('l = ',l)
    # ITERATE COUNTER
'''
'''
for l in range(0,im_t_steps+1):
    # k1 CALCULATION
    KE = (2/r)*(Dr @ phi) + Dr2 @ phi
    # HAMILTONIAN TERMS
    H_KE = -0.5*KE # KE term
    H_LHY= LHY_coef*abs(phi)**3*phi # LHY term
    H_int = int_coef*abs(phi)**2*phi # s-wave term
    H_trap = V*phi # potential term

    k1 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*phi)
    
    # k2 CALCULATION
    KE = (2/r)*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r)*(Dr @ k1) + Dr2 @ k1) 
    # HAMILTONIAN TERMS
    H_KE = -0.5*KE # KE term
    H_LHY = LHY_coef*abs(phi + k1/2)**3*(phi + k1/2) # LHY term
    H_int = int_coef*abs(phi + k1/2)**2*(phi + k1/2) # s-wave term
    H_trap = V*(phi + k1/2) # potential term
    
    k2 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*(phi + k1/2))
    
    # k3 CALCULATION
    KE = (2/r)*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r)*(Dr @ k2) + Dr2 @ k2)  
    # HAMILTONIAN TERMS 
    H_KE = -0.5*KE # KE term
    H_LHY = LHY_coef*abs(phi + k2/2)**3*(phi + k2/2) # LHY term
    H_int = int_coef*abs(phi + k2/2)**2*(phi + k2/2) # s-wave term
    H_trap = V*(phi + k2/2) # potential term
   
    k3 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*(phi + k2/2))
    
    #k4 CALCULATION
    KE = (2/r)*(Dr @ phi) + Dr2 @ phi + ((2/r)*(Dr @ k3) + Dr2 @ k3)  
    # HAMILTONIAN TERMS
    H_KE = -0.5*KE # KE term
    H_LHY = LHY_coef*abs(phi + k3)**3*(phi + k3) # LHY term
    H_int = int_coef*abs(phi + k3)**2*(phi + k3) # s-wave term
    H_trap = V*(phi + k3) # potential term
    
    k4 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*(phi + k3))
    
    # FINAL RUNGE-KUTTA STEP    
    phi = phi + (1./6)*(k1 + 2*k2 + 2*k3 + k4)
    
    # NEUMANN BOUNDARY CONDITIONS
    # phi(j+1) - phi(j-1) = 0
    phi[0] = phi[2]
    phi[-1] = phi[-3]

    # phi(j+1) - phi(j) = 0
    #phi[0] = phi[1]
    #phi[-1] = phi[-2]

    # WAVEFUNCTION NORMALISED
    Norm = 4*pi*np.trapz(r**2*abs(phi)**2)*dr
    phi = phi/np.sqrt(Norm) 
    
    # MU AND TOLERANCE CALCULATION
    mu_old = mu
    phi_r = np.gradient(phi,dr)
    mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2 + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
    tol = abs((mu-mu_old)/mu_old)
    if (l % 10000 == 0):
        plt.plot(r,np.sqrt(N)*phi)
        plt.xlim((0,Lr))
        plt.ylim((0,2))
        plt.xlabel("$r$")
        plt.ylabel("$\phi_0$")
        plt.title("$N = $"+str(N))
        plt.pause(0.2)
        print('l = ',l)
    
    #if (l % 10000 == 0):
    #    with open ('N_19_dens.txt','a') as f:
    #        f.write(str(t) + ' ' + str(max(np.sqrt(N)*phi)) + '\n')
    #    print(max(np.sqrt(N)*phi))
    t_array[l] = t
    mean_r2[l] = max(phi)
    phi_dr2[l] = phi[1]
    phi_3dr2[l] = phi[2]
    # ITERATE TIME     
    t = t + dt    
    # ITERATE COUNTER
    count = count + 1


#plt.plot(t_array,mean_r2)
plt.plot(t_array,phi_dr2)
plt.plot(t_array,phi_3dr2)
plt.legend(("dr/2","3*dr/2"))

###############################################################################
'''
# REAL TIME

# INITIALISATION OF HAMILTONIAN ARRAYS
H_KE = np.zeros(phi.size).astype(complex)
H_trap = np.zeros(phi.size).astype(complex)
H_LHY = np.zeros(phi.size).astype(complex)
H_int = np.zeros(phi.size).astype(complex)
KE = np.zeros(phi.size).astype(complex)

# INITIALISATION OF TIME-STEPPING AND DATA SAVING ARRAYS
spacetime = np.empty(r.size).astype(complex) # array to save density snapshots
t = 0 # initial time 
t_array = np.zeros(t_steps+1) 
mean_r2 = np.zeros(t_steps+1) # observable used here <r^2> 

# COEFFICIENTS FOR REAL TIME
if int_gas == 0:
    int_coef = 0
    LHY_coef = 0
elif int_gas == 1:    
    int_coef = -3*N
    LHY_coef = (5/2)*N**(3/2)    

# DIFFERENTIAL OPERATORS
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
# 1st order derivative
d11 = np.zeros(r.size-1); d12 = -1*np.ones(r.size); d13 = np.ones(r.size-1)
Dr = (1/dr)*tridiag(d11,d12,d13)
# 2nd order derivative
d21 = np.ones(r.size-1); d22 = -2*np.ones(r.size); d23 = np.ones(r.size-1)
Dr2 = (1/dr**2)*tridiag(d21, d22, d23)

if method == 0:
    # EULER REAL TIME
    for l in range(0,t_steps+1):
        # CALCULATING THE DERVATIVES FOR THE KINETIC ENERGY TERM
        for j in range(1,phi.size-1):
            KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2    
        # SETTING EACH TERM IN THE HAMILTONIAN
        H_KE = -0.5*KE
        H_trap = V*phi
        H_LHY = LHY_coef*abs(phi)**3*phi
        H_int = int_coef*abs(phi)**2*phi
        # COMPLETING THE EULER TIME STEP
        phi = phi - 1j*dt*(H_KE + H_trap + H_LHY + H_int)
        # NEUMANN BOUNDARY CONDITIONS
        phi[0] = phi[1]
        phi[-1] = phi[-2]

        #PLOT WAVEFUNCTION 
        if l%10==0:
            plt.plot(r,N*abs(phi)**2)
            plt.xlim(0,10)
            plt.pause(0.4)
            
elif method == 1:
    # RK4 REAL TIME
    for l in range(0,t_steps+1):
        # k1 CALCULATION
        KE = (2/r)*np.matmul(Dr,phi) + np.matmul(Dr2,phi) 
        # HAMILTONIAN TERMS    
        H_KE = -0.5*KE # kinetic energy term  
        H_LHY = LHY_coef*abs(phi)**3*phi # LHY term
        H_int = int_coef*abs(phi)**2*phi # s-wave term
        H_trap = V*phi # potential term 
        
        k1 = -1j*dt*(H_KE + H_trap + H_LHY + H_int ) 
        
        # k2 CALCULATION
        KE = (2/r)*np.matmul(Dr,phi) + np.matmul(Dr2,phi) + 0.5*((2/r)*np.matmul(Dr,k1) + np.matmul(Dr2,k1))         
        # HAMILTONIAN TERMS
        H_KE = -0.5*KE # kinetic energy term
        H_LHY = LHY_coef*abs(phi + k1/2)**3*(phi + k1/2) # LHY term
        H_int = int_coef*abs(phi + k1/2)**2*(phi + k1/2) # s-wave term
        H_trap = V*(phi + k1/2) # potential term
        
        k2 = -1j*dt*(H_KE + H_trap + H_LHY + H_int)
        
        # k3 CALCULATION
        KE = (2/r)*np.matmul(Dr,phi) + np.matmul(Dr2,phi) + 0.5*((2/r)*np.matmul(Dr,k2) + np.matmul(Dr2,k2))  
        # HAMILTONIAN TERMS
        H_KE = -0.5*KE # kinetic energy term
        H_LHY = LHY_coef*abs(phi + k2/2)**3*(phi + k2/2) # LHY term
        H_int = int_coef*abs(phi + k2/2)**2*(phi + k2/2) # s-wave term
        H_trap = V*(phi + dr*k2/2) # potential term
        
        k3 = -1j*dt*(H_KE + H_trap + H_LHY + H_int)

        #k4 CALCULATION
        KE = (2/r)*np.matmul(Dr,phi) + np.matmul(Dr2,phi) + ((2/r)*np.matmul(Dr,k3) + np.matmul(Dr2,k3))  
        # HAMILTONIAN TERMS
        H_KE = -0.5*KE # kinetic energy term
        H_LHY = LHY_coef*abs(phi + k3)**3*(phi + k3) # LHY term
        H_int = int_coef*abs(phi + k3)**2*(phi + k3) # s-wave term
        H_trap = V*(phi + k3) # potential term
        
        k4 = -1j*dt*(H_KE + H_trap + H_LHY + H_int)

        # FINAL RUNGE-KUTTA STEP    
        phi = phi + (1./6)*(k1 + 2*k2 + 2*k3 + k4)
       
        # NEUMANN BOUNDARY CONDITIONS
        # phi(j+1) - phi(j-1) = 0
        #phi[0] = phi[2]
        #phi[-1] = phi[-3]

        # phi(j+1) - phi(j) = 0
        phi[0] = phi[1]
        phi[-1] = phi[-2]
        
        # SAVING DATA
        t_array[l] = t # save current time
        spacetime = np.vstack((spacetime,phi)) # save current wavefunction
        mean_r2[l] = 4*pi*np.trapz(r**4*abs(phi)**2)*dr # save observable <r^2>

        #PLOT WAVEFUNCTION 
        if l%1000==0:
            print(l)
            plt.plot(r,N*abs(phi)**2)
            #plt.plot(r,np.angle(phi))
            plt.xlim(0,10)
            #plt.pause(0.4)
            
        # ITERATE TIME     
        t = t + dt
'''
###############################################################################
# PLOTS:
'''
plt.pcolor(N*abs(spacetime)**2)
plt.plot(r,np.sqrt(N)*phi)
plt.xlim((0,10))
plt.ylim((0,0.1))
plt.xlabel("$r$")
plt.ylabel("$\phi_0$")

# Plotting the observable <r^2> as a function of time
plt.plot(t_array,mean_r2)
plt.show()

# Attempted signal processing on the observable <r^2>
w_array = scipy.fftpack.fft(np.fft.fftshift(t_array))
r2_w = scipy.fftpack.fft(np.fft.fftshift(mean_r2))
plt.plot(abs(w_array),abs(r2_w))
plt.plot(abs(r2_w))
'''
###############################################################################

# xi and tau finder
#hbar = 1.0545718*1e-34
#a_0 = 5.29*1e-11
#a_11 = 84.3*a_0
#a_22 = 33.5*a_0
#a_12 = -0.25*12.09*a_0 - np.sqrt(a_11*a_22)
#m_1 = 39*1.66053904020*1e-27
#m_2 = 39*1.66053904020*1e-27
#[xi,tau,dg] = xi_tau_finder(a_11,a_22,a_12,m_1,m_2)

# PARAMETERS
#g_11 = hbar**2*4*pi*a_11/m_1
#g_22 = hbar**2*4*pi*a_22/m_2
#g_12 = hbar**2*2*pi*a_12*(m_1+m_2)/(m_1*m_2)
#dg = g_12 + np.sqrt(g_11*g_22)
   
# MINARDI'S CONSTANTS
#z = m_2/m_1
#u = 1
#x = np.sqrt(g_22/g_11)

#if m_1 == m_2:
#    n_01 = (25*pi/1024)*(a_12 + np.sqrt(a_11*a_22))**2/(a_11*a_22*np.sqrt(a_11)*(np.sqrt(a_11) + np.sqrt(a_22))**5)
#else:
    # LAMBDA DEFINE F
#    F_int = integrate.quad(lambda k: k**2*(np.sqrt(0.5*(k**2*(1 + (x/z)) + 
#            0.25*k**4*(1 +(1/z**2))) + np.sqrt(0.25*((k**2 + 0.25*k**4) - 
#            ((x*k**2)/z + (k**4)/(4*z**2)))**2 + u*x*k**4/z)) +
#            np.sqrt(0.5*(k**2*(1+(x/z)) + 0.25*k**4*(1+(1/z**2))) -
#            np.sqrt(0.25*((k**2 + 0.25*k**4) - ((x*k**2)/z + 
#            (k**4)/(4*z**2)))**2 + u*x*k**4/z)) - (1+z)*k**2/(2*z) -
#            (1+x) + (1/k**2)*(1 + x**2*z + 4*u*x*z/(1+z))),0,np.inf)
    # CALCULATION OF f
#    f = (15/32)*F_int
    # CALCULATION OF n_01
#    n_01 = (25*pi/1024)*(1/f**2)*(1/a_11**3)*(dg**2/(g_11*g_22))
    
#n_02 = n_01*np.sqrt(g_11/g_22) 
# XI CALCULATION
#xi = np.sqrt(1.5*(np.sqrt(g_22)/m_1 + np.sqrt(g_11)/m_2)/(abs(dg)*np.sqrt(g_11)*n_01))
    
# TAU CALCULATION
#tau = 1.5*(np.sqrt(g_11) + np.sqrt(g_22))/(abs(dg)*np.sqrt(g_11)*n_01)