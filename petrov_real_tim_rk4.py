# PACKAGES
import numpy as np
from scipy.sparse import eye
pi = np.math.pi

def petrov_real_tim_rk4_mat(phi,mu,r,dr,dt,N,V,int_gas,t_steps,mode):
    # GPE COEFFICIENTS
    if int_gas == 0:
        int_coef = 0
        LHY_coef = 0
    elif int_gas == 1:
        int_coef = -3*N
        LHY_coef = (5/2)*N**(3/2)

    # DIFFERENTIAL OPERATORS
    # first order derivative in the form of a sparse matrix (centrally defined)
    Dr = (1/(2*dr))*(-1*eye(phi.size-2,phi.size,k=0,dtype=float) + eye(phi.size-2,phi.size,k=2,dtype=float))
    # second order derivative in the form of a sparse matrix (centrally defined 3-point formula)
    Dr2 =  (1/dr**2)*(eye(phi.size-2,phi.size,k=0,dtype=float) -2*eye(phi.size-2,phi.size,k=1,dtype=float) + eye(phi.size-2,phi.size,k=2,dtype=float))

    # INITIALISING ARRAYS
    H_KE = np.zeros(phi.size).astype(complex)
    H_LHY = np.zeros(phi.size).astype(complex)
    H_int = np.zeros(phi.size).astype(complex)
    H_trap = np.zeros(phi.size).astype(complex)
    KE = np.zeros(phi.size).astype(complex)
    k1 = np.zeros(phi.size).astype(complex)
    k2 = np.zeros(phi.size).astype(complex)
    k3 = np.zeros(phi.size).astype(complex)
    k4 = np.zeros(phi.size).astype(complex)
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
    if mode == 1:
        lamb = 1e-4 # small constant
        phi = np.exp(1j*lamb*r**2)*phi # small phase imprint of the form exp(i*lambda*F) where F = r^2 for breathing mode

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

        # SAVING DATA AND OBSERVABLES
        if (l % t_save == 0):
            spacetime[:,l//t_save] = phi # save current wavefunction
            phase[:,l//t_save] = np.angle(phi) # save current phase
            t_array[l//t_save] = t # save current time
            mean_r2[l//t_save] = 4*pi*np.trapz(r**4*np.abs(phi)**2)*dr # save current observable <r^2>

        # ITERATE TIME     
        t = t + dt   
        
        # ITERATE COUNTER
        count = count + 1

    return phi,spacetime,t_array,mean_r2

