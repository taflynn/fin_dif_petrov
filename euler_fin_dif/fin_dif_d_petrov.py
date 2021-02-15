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
Nr = 1024
dr = Lr/Nr
r = np.arange(-1/2,(Nr + 3/2),1)*dr

# TIME STEP
dt = 0.4*dr**2

# PARAMETERS
N = 500

# GPE COEFFICIENTS
int_coef= -3*N
LHY_coef = (5/2)*N**(3/2)

# INITIALISE TOLERANCE AND WAVEFUNCTION
phi_temp = np.zeros(r.size)
phi_0 = np.exp(-(r)**2/(2*(1)**2))
#phi_0 = np.tanh(r-Lr)**2
Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
#Norm = np.trapz(abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm)
phi = phi_0
tol = 1

# INITIAL MU
phi_r = np.gradient(phi,dr)
mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
#mu = np.trapz(0.5*abs(phi_r)**2 + V*abs(phi)**2 + int_coef*abs(phi)**4)/np.trapz(abs(phi)**2)

H_KE = np.zeros(phi.size)
H_LHY = np.zeros(phi.size)
H_int = np.zeros(phi.size)
KE = np.zeros(phi.size)

count = 1

while tol>1e-9:
    
    for j in range(1,phi.size-1):
        KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2  
        
    H_KE = -0.5*KE
    H_LHY = LHY_coef*abs(phi)**3*phi
    H_int = int_coef*abs(phi)**2*phi
    
    #for j in range(1,phi.size-2):
    #    phi_temp[j] = phi[j] + (dt/2)*((phi[j+1] - 2*phi[j] + phi[j-1])/dr**2) - int_coef*dt*abs(phi[j])**2*phi[j] - LHY_coef*abs(phi[j])**3*phi[j] #- mu*dt*phi[j] 
    #phi = phi_temp 
    
    phi = phi - dt*(H_KE + H_LHY + H_int)

    # DIRICHLET BOUNDARY CONDITIONS
    
    #phi[0] = 0;
    #phi[-1] = 0;
   
    # NEUMANN BOUNDARY CONDITIONS

    phi[0] = phi[2]
    phi[-1] = phi[-3]


    # WAVEFUNCTION NORMALISED
    Norm = 4*pi*np.trapz(r**2*abs(phi)**2)*dr
    #Norm = np.trapz(abs(phi)**2)*dr
    phi = phi/np.sqrt(Norm) 
    
    # MU AND TOLERANCE CALCULATION
    mu_old = mu
    phi_r = np.gradient(phi,dr)
    #mu = np.trapz(0.5*abs(phi_r)**2 + V*abs(phi)**2 + int_coef*abs(phi)**4)/np.trapz(abs(phi)**2)
    mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
    tol = abs((mu-mu_old)/mu_old)

    #count = count + 1
    #if count%1000 == 0:
    #    plt.plot(r,phi)
    #    plt.pause(0.01)

# SAVING DATA 
np.savetxt('N_'+str(N)+'_phi.txt',(r,np.sqrt(N)*phi),fmt='%.10e')

# PLOTTING
#set path
fig = plt.figure(figsize = (15.,10.))
plt.plot(r,np.sqrt(N)*phi)
plt.xlim(0,12)
plt.ylim(0,1.2)
plt.xlabel("$r$")
plt.ylabel("$\phi_0$")
plt.title('N ='+ str(N))
fig.tight_layout()
fig.savefig(fname = 'N_'+str(N)+'_phi.png',dpi=300)
#plt.savefig(fname = 'N_'+str(N)+'_phi.pdf',bbox_inches='tight')
#plt.show()