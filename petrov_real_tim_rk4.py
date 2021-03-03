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
    Dr = (1/(2*dr))*(-1*eye(phi.size-2,phi.size,k=0,dtype=float) + eye(phi.size-2,phi.size,k=2,dtype=float))
    Dr2 =  (1/dr**2)*(eye(phi.size-2,phi.size,k=0,dtype=float) -2*eye(phi.size-2,phi.size,k=1,dtype=float) + eye(phi.size-2,phi.size,k=2,dtype=float))

    # initialising matrices
    H_KE = np.zeros(phi.size).astype(complex)
    H_LHY = np.zeros(phi.size).astype(complex)
    H_int = np.zeros(phi.size).astype(complex)
    H_trap = np.zeros(phi.size).astype(complex)
    KE = np.zeros(phi.size).astype(complex)
    k1 = np.zeros(phi.size).astype(complex)
    k2 = np.zeros(phi.size).astype(complex)
    k3 = np.zeros(phi.size).astype(complex)
    k4 = np.zeros(phi.size).astype(complex)
    phi = phi.astype(complex)
    t_save = 100
    t = 0
    spacetime = np.zeros((r.size,(t_steps//t_save))).astype(complex)
    phase = np.zeros((r.size,(t_steps//t_save))).astype(complex)
    t_array = np.zeros((t_steps//t_save)) 
    mean_r2 = np.zeros((t_steps//t_save)) # observable used here <r^2> 
    # swap to a smaller time step in real time
    dt = 0.1*dr**2

    # invoke breathing mode
    lamb = 1e-4
    phi = np.exp(1j*lamb*r**2)*phi

    for l in range(0,t_steps):  
        # k1 CALCULATION
        KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi
        # HAMILTONIAN TERMS
        H_KE[1:-1] = -0.5*KE[1:-1] # KE term
        H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1])**3*phi[1:-1] # LHY term
        H_int[1:-1] = int_coef*np.abs(phi[1:-1])**2*phi[1:-1] # s-wave term
        H_trap[1:-1] = V[1:-1]*phi[1:-1] # potential term

        k1[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*phi[1:-1])

        k1[0] = k1[1]
        k1[-1] = k1[-2]

        # k2 CALCULATION
        KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k1) + Dr2 @ k1) 
        # HAMILTONIAN TERMS
        H_KE[1:-1] = -0.5*KE[1:-1] # KE term
        H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k1[1:-1]/2)**3*(phi[1:-1] + k1[1:-1]/2) # LHY term
        H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k1[1:-1]/2)**2*(phi[1:-1] + k1[1:-1]/2) # s-wave term
        H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k1[1:-1]/2) # potential term

        k2[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k1[1:-1]/2))

        k2[0] = k2[1]
        k2[-1] = k2[-2]

        # k3 CALCULATION
        KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k2) + Dr2 @ k2)  
        # HAMILTONIAN TERMS 
        H_KE[1:-1] = -0.5*KE[1:-1] # KE term
        H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k2[1:-1]/2)**3*(phi[1:-1] + k2[1:-1]/2) # LHY term
        H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k2[1:-1]/2)**2*(phi[1:-1] + k2[1:-1]/2) # s-wave term
        H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k2[1:-1]/2) # potential term

        k3[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k2[1:-1]/2))

        k3[0] = k3[1]
        k3[-1] = k3[-2]

        # k4 CALCULATION
        KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + ((2/r[1:-1])*(Dr @ k3) + Dr2 @ k3)  
        # HAMILTONIAN TERMS
        H_KE[1:-1] = -0.5*KE[1:-1] # KE term
        H_LHY[1:-1] = LHY_coef*np.abs(phi[1:-1] + k3[1:-1])**3*(phi[1:-1] + k3[1:-1]) # LHY term
        H_int[1:-1] = int_coef*np.abs(phi[1:-1] + k3[1:-1])**2*(phi[1:-1] + k3[1:-1]) # s-wave term
        H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k3[1:-1]) # potential term

        k4[1:-1] = -1j*dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k3[1:-1]))

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
            phase[:,l//t_save] = np.angle(phi)
            t_array[l//t_save] = t
            mean_r2[l//t_save] = 4*pi*np.trapz(r**4*np.abs(phi)**2)*dr

        # ITERATE TIME     
        t = t + dt   

        # ITERATE COUNTER
        count = count + 1

    return phi,spacetime,t_array,mean_r2

