import numpy as np

import matplotlib.pyplot as plt


pi = np.math.pi

# GRID
Lr = 10
Nr = 256
dr = Lr/Nr
r = np.arange(-3/2,(Nr + 5/2),1)*dr
# TIME STEP
dt = 0.4*dr**2

# PARAMETERS
N = 500

V = np.zeros(r.size)
# INITIALISE TOLERANCE AND WAVEFUNCTION
phi_0 = np.exp(-(r)**2/(2*(1)**2))
Norm = 4*pi*np.trapz(r**2*abs(phi_0)**2)*dr
phi_0 = phi_0/np.sqrt(Norm)
phi = phi_0

# GPE COEFFICIENTS
int_coef = -3*N
LHY_coef = (5/2)*N**(3/2)

# INITIAL MU
phi_r = np.gradient(phi,dr)
mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2  + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
tol = 1

# DIFFERENTIAL OPERATORS
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

d11 = np.zeros(r.size-1); d12 = -1*np.ones(r.size-0); d13 = np.ones(r.size-1)
Dr = (1/dr)*tridiag(d11,d12,d13)

d21 = np.ones(r.size-1); d22 = -2*np.ones(r.size-0); d23 = np.ones(r.size-1)
Dr2 = (1/dr**2)*tridiag(d21, d22, d23)
    
# INITIALISING ARRAYS
H_KE = np.zeros(phi.size)
H_LHY = np.zeros(phi.size)
H_int = np.zeros(phi.size)
H_trap = np.zeros(phi.size)
KE = np.zeros(phi.size)

count = 1

while tol>1e-9:
    # k1 CALCULATION
    #KE = (2/r[2:-2])*np.matmul(Dr,phi[2:-2]) + (1/r[2:-2]**2)*np.matmul(Dr2,phi[2:-2])

    #H_KE = -0.5*KE
    #H_LHY= LHY_coef*abs(phi[2:-2])**3*phi[2:-2]
    #H_int = int_coef*abs(phi[2:-2])**2*phi[2:-2]
    #H_trap = V[2:-2]*phi[2:-2]
    
    #k1 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)
    
    # k2 CALCULATION
    #KE = (2/r[2:-2])*np.matmul(Dr,phi[2:-2]) + (1/r[2:-2]**2)*np.matmul(Dr2,phi[2:-2]) + 0.5*(2/r[2:-2])*np.matmul(Dr,k1) + 0.5*(1/r[2:-2]**2)*np.matmul(Dr2,k1) 
    
    #H_KE = -0.5*KE
    #H_LHY = LHY_coef*abs(phi[2:-2] + k1/2)**3*(phi[2:-2] + k1/2)
    #H_int = int_coef*abs(phi[2:-2] + k1/2)**2*(phi[2:-2] + k1/2)
    #H_trap = V[2:-2]*(phi[2:-2] + k1/2)
    
    #k2 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)
      
    # k3 CALCULATION
    #KE = (2/r[2:-2])*np.matmul(Dr,phi[2:-2]) + (1/r[2:-2]**2)*np.matmul(Dr2,phi[2:-2]) + 0.5*(2/r[2:-2])*np.matmul(Dr,k2) + 0.5*(1/r[2:-2]**2)*np.matmul(Dr2,k2)  
    
    #H_KE = -0.5*KE
    #H_LHY = LHY_coef*abs(phi[2:-2] + k2/2)**3*(phi[2:-2] + k2/2)
    #H_int = int_coef*abs(phi[2:-2] + k2/2)**2*(phi[2:-2] + k2/2)
    #H_trap = V[2:-2]*(phi[2:-2] + k2/2)
    
    #k3 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)

    #k4 CALCULATION
    #KE = (2/r[2:-2])*np.matmul(Dr,phi[2:-2]) + (1/r[2:-2]**2)*np.matmul(Dr2,phi[2:-2]) + (2/r[2:-2])*np.matmul(Dr,k3) + (1/r[2:-2]**2)*np.matmul(Dr2,k3)  
    
    #H_KE = -0.5*KE
    #H_LHY = LHY_coef*abs(phi[2:-2] + k3)**3*(phi[2:-2] + k3)
    #H_int = int_coef*abs(phi[2:-2] + k3)**2*(phi[2:-2] + k3)
    #H_trap = V[2:-2]*(phi[2:-2] + k3)
    
    #k4 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)
    
    # FINAL RUNGE-KUTTA STEP    
    #phi[2:-2] = phi[2:-2] + (1./6)*(k1 + 2*k2 + 2*k3 + k4)
       
    # k1 CALCULATION
    KE = (2/r)*np.matmul(Dr,phi) + np.matmul(Dr2,phi)

    H_KE = -0.5*KE
    H_LHY= LHY_coef*abs(phi)**3*phi
    H_int = int_coef*abs(phi)**2*phi
    H_trap = V*phi
    
    k1 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)
    
    # k2 CALCULATION
    KE = (2/r)*np.matmul(Dr,phi) + np.matmul(Dr2,phi) + 0.5*(2/r)*np.matmul(Dr,k1) + 0.5*np.matmul(Dr2,k1) 
    
    H_KE = -0.5*KE
    H_LHY = LHY_coef*abs(phi + k1/2)**3*(phi + k1/2)
    H_int = int_coef*abs(phi + k1/2)**2*(phi + k1/2)
    H_trap = V*(phi + k1/2)
    
    k2 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)
      
    # k3 CALCULATION
    KE = (2/r)*np.matmul(Dr,phi) + np.matmul(Dr2,phi) + 0.5*(2/r)*np.matmul(Dr,k2) + 0.5*np.matmul(Dr2,k2)  
    
    H_KE = -0.5*KE
    H_LHY = LHY_coef*abs(phi + k2/2)**3*(phi + k2/2)
    H_int = int_coef*abs(phi + k2/2)**2*(phi + k2/2)
    H_trap = V*(phi + k2/2)
    
    k3 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)

    #k4 CALCULATION
    KE = (2/r)*np.matmul(Dr,phi) + np.matmul(Dr2,phi) + (2/r)*np.matmul(Dr,k3) + np.matmul(Dr2,k3)  
    
    H_KE = -0.5*KE
    H_LHY = LHY_coef*abs(phi + k3)**3*(phi + k3)
    H_int = int_coef*abs(phi + k3)**2*(phi + k3)
    H_trap = V*(phi + k3)
    
    k4 = -dr*dt*(H_KE + H_trap + H_LHY + H_int)
    
    # FINAL RUNGE-KUTTA STEP    
    phi = phi + (1./6)*(k1 + 2*k2 + 2*k3 + k4)
       
    
    # NEUMANN BOUNDARY CONDITIONS

    #phi[0] = phi[3]
    #phi[1] = phi[2]
    #phi[-1] = phi[-4]
    #phi[-2] = phi[-3]

    phi[1] = phi[2]
    phi[0] = phi[1]
    phi[-2] = phi[-3]
    phi[-1] = phi[-2]

    # WAVEFUNCTION NORMALISED
    Norm = 4*pi*np.trapz(r**2*abs(phi)**2)*dr
    phi = phi/np.sqrt(Norm) 
    
    # MU AND TOLERANCE CALCULATION
    mu_old = mu
    phi_r = np.gradient(phi,dr)
    mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2 + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
    tol = abs((mu-mu_old)/mu_old)
    if count%100==0:
        plt.plot(r,np.sqrt(N)*phi)
        plt.pause(0.4)
    count = count + 1
    
plt.plot(r,phi)