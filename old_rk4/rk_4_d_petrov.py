# PACKAGES
import timeit

import os

import numpy as np

import matplotlib.pyplot as plt

os.chdir('C:\\Users\\TAFly\\Documents\\PhD\\1st_year\\Python_GPE\\fin_dif_codes')

timeit.timeit()

pi = np.math.pi

# GRID
Lr = 20
Nr = 256
dr = Lr/Nr
r = np.arange(-3/2,(Nr + 5/2),1)*dr

# TIME STEP
dt = 0.4*dr**2

# TRAP? (NO = 0, YES = 1)
trap = 0

# PARAMETERS
N = 10

# GPE COEFFICIENTS
int_coef = -3*N
LHY_coef = (5/2)*N**(3/2)
if trap == 0:
    V = np.zeros(r.size)
elif trap == 1:
    V = 0.5*r**2

# INITIALISE TOLERANCE AND WAVEFUNCTION
phi_0 = np.exp(-(r)**2/(2*(1)**2))
Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm)
phi = phi_0
tol = 1

# INITIAL MU
phi_r = np.gradient(phi,dr)
mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2  + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)

H_KE = np.zeros(phi.size)
H_LHY = np.zeros(phi.size)
H_int = np.zeros(phi.size)
H_trap = np.zeros(phi.size)
KE = np.zeros(phi.size)

count = 1

while tol>1e-9:
    # k1 CALCULATION
    for j in range(1,phi.size-1):
        KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2  
        
    H_KE = -0.5*KE
    H_LHY = LHY_coef*abs(phi)**3*phi
    H_int = int_coef*abs(phi)**2*phi
    H_trap = V*phi
    
    k1 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)
    
    # k2 CALCULATION
    for j in range(1,phi.size-1):
        KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2 + 0.5*(2/r[j])*(k1[j+1] - k1[j])/dr + 0.5*(k1[j+1] - 2*k1[j] + k1[j-1])/dr**2  
    
    H_KE = -0.5*KE
    H_LHY = LHY_coef*abs(phi + k1/2)**3*(phi + k1/2)
    H_int = int_coef*abs(phi + k1/2)**2*(phi + k1/2)
    H_trap = V*(phi + k1/2)
    
    k2 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)

   
    # k3 CALCULATION

    for j in range(1,phi.size-1):
        KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2 + 0.5*(2/r[j])*(k2[j+1] -k2[j])/dr + 0.5*(k2[j+1] - 2*k2[j] + k2[j-1])/dr**2  
    
    H_KE = -0.5*KE
    H_LHY = LHY_coef*abs(phi + k2/2)**3*(phi + k2/2)
    H_int = int_coef*abs(phi + k2/2)**2*(phi + k2/2)
    H_trap = V*(phi + k2/2)
    
    k3 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)

    #k4 CALCULATION

    for j in range(1,phi.size-1):
        KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2 + (2/r[j])*(k3[j+1] -k3[j])/dr + (k3[j+1] - 2*k3[j] + k3[j-1])/dr**2  
    
    H_KE = -0.5*KE
    H_LHY = LHY_coef*abs(phi + k3)**3*(phi + k3)
    H_int = int_coef*abs(phi + k3)**2*(phi + k3)
    H_trap = V*(phi + k3)
    
    k4 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)

    # FINAL RUNGE-KUTTA STEP    
    phi = phi + (1./6)*(k1 + 2*k2 + 2*k3 + k4)
   
    # NEUMANN BOUNDARY CONDITIONS

    phi[0] = phi[3]
    phi[1] = phi[2]
    phi[-1] = phi[-4]
    phi[-2] = phi[-3]

    # WAVEFUNCTION NORMALISED
    Norm = 4*pi*np.trapz(r**2*abs(phi)**2)*dr
    phi = phi/np.sqrt(Norm) 
    
    # MU AND TOLERANCE CALCULATION
    mu_old = mu
    phi_r = np.gradient(phi,dr)
    mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2 + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
    tol = abs((mu-mu_old)/mu_old)

    count = count + 1
    if count%10000 == 0:
        plt.plot(r,np.sqrt(N)*phi)
        plt.pause(0.01)

# PLOTTING
#phi_tf = ((mu - 0.5*r**2)/int_coef)*np.heaviside((mu - 0.5*r**2)/int_coef,r)
#set path
fig = plt.figure(figsize = (15.,10.))
plt.plot(r,np.sqrt(N)*phi)
#plt.plot(r,N*phi_tf)
plt.xlim(0,12)
plt.ylim(0,1.5)
plt.xlabel("$r$")
plt.ylabel("$\phi_0$")
#plt.title('N ='+ str(N))
fig.tight_layout()
#fig.savefig(fname = 'N_'+str(N)+'_phi.png',dpi=300)
#plt.savefig(fname = 'N_'+str(N)+'_phi.pdf',bbox_inches='tight')
#plt.show()
