# PACKAGES
import numpy as np

import matplotlib.pyplot as plt

pi = np.math.pi

# GRID
Lr = 20
Nr = 1024   
dr = Lr/Nr
dk = 2*pi/Lr
r = np.arange(-(Nr-1/2),(Nr+1/2),1)*dr

# TIME STEP
dt = 0.4*dr**2

# GPE COEFFICIENTS
A = 500
V = 0.5*r**2

# INITIALISE TOLERANCE AND WAVEFUNCTION
psi_0 = np.exp(-r**2/(2*(1)**2))
Norm = np.trapz(abs(psi_0)**2)*dr
psi_0 = psi_0/np.sqrt(Norm)
psi = psi_0
tol = 1

sec_dif = np.zeros(psi.size)

# INITIAL MU
psi_r = np.gradient(psi,dr)

mu = np.trapz(0.5*abs(psi_r)**2 + V*abs(psi)**2 + A*abs(psi)**4)/np.trapz(abs(psi)**2)

H_KE = np.zeros(psi.size)
H_trap = np.zeros(psi.size)
H_int = np.zeros(psi.size)

count = 1

while tol>1e-9:
    
    for j in range(1,psi.size-1):
        sec_dif[j] = (psi[j+1]- 2*psi[j] + psi[j-1])/dr**2  
        
    H_KE = -0.5*sec_dif
    H_trap = V*psi
    H_int = A*abs(psi)**2*psi
    
    psi = psi - dt*(H_KE + H_trap + H_int)
    
    # DIRICHLET BOUNDARY CONDITIONS
    
    psi[0] = 0;
    psi[-1] = 0;
    
    # WAVEFUNCTION NORMALISED
    Norm = np.trapz(abs(psi)**2)*dr
    psi = psi/np.sqrt(Norm) 
    
    # MU AND TOLERANCE CALCULATION
    mu_old = mu
    psi_r = np.gradient(psi,dr)
    mu = np.trapz(0.5*abs(psi_r)**2 + V*abs(psi)**2 + A*abs(psi)**4)/np.trapz(abs(psi)**2)
    tol = abs((mu-mu_old)/mu_old)

    count = count + 1

for l in range(0,1000):
    for j in range(1,psi.size-1):
        sec_dif[j] = (psi[j+1]- 2*psi[j] + psi[j-1])/dr**2  
        
    H_KE = -0.5*sec_dif
    H_trap = V*psi
    H_int = A*abs(psi)**2*psi
    
    psi = psi +  1j*dt*(H_KE + H_trap + H_int)
    
    # DIRICHLET BOUNDARY CONDITIONS
    
    psi[0] = 0;
    psi[-1] = 0;
    
    plt.plot(r,abs(psi)**2)
    plt.pause(0.4)

# PLOTTING
tf_dens = ((mu - 0.5*r**2)/A)*np.heaviside((mu - 0.5*r**2)/A,r)

plt.plot(r,abs(psi)**2,label="$n_0$")
plt.plot(r,tf_dens,label="$n_{TF}$")

plt.xlim(-Lr/2, Lr/2)
plt.legend(loc='upper left')
plt.xlabel("$z$")
plt.ylabel("$|\psi|^2$")
plt.title("$\beta = $",A)
