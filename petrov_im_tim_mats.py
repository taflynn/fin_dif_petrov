# PACKAGES
import numpy as np

from scipy.sparse import diags

pi = np.math.pi

def petrov_im_tim_rk4(phi,r,dr,dt,N,V):
    # GPE COEFFICIENTS
    int_coef = -3*N
    LHY_coef = (5/2)*N**(3/2)

    # INITIAL MU
    phi_r = np.gradient(phi,dr)
    mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2  + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
    tol = 1

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
        count = count + 1
    return(phi,mu)

def mat_H_petrov_im_tim_rk4(phi,r,dr,dt,N,V):
    # GPE COEFFICIENTS
    int_coef = 0#-3*N
    LHY_coef = 0#(5/2)*N**(3/2)

    # INITIAL MU
    phi_r = np.gradient(phi,dr)
    mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2  + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
    tol = 1

    # DIFFERENTIAL OPERATORS
    #def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    #    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

    #d11 = np.zeros(r.size-1); d12 = -1*np.ones(r.size); d13 = np.ones(r.size-1)
    #Dr = (1/dr)*tridiag(d11,d12,d13)

    #d21 = np.ones(r.size-1); d22 = -2*np.ones(r.size); d23 = np.ones(r.size-1)
    #Dr2 = (1/dr**2)*tridiag(d21, d22, d23)
    
    # INITIALISING ARRAYS
    #H_KE = np.zeros(phi.size)
    #H_LHY = np.zeros(phi.size)
    #H_int = np.zeros(phi.size)
    #H_trap = np.zeros(phi.size)
    #KE = np.zeros(phi.size)

    count = 1
    while tol>1e-9:
        # k1 CALCULATION
        H_KE_diag1 = [-1*np.ones(r.size)/r,np.ones(r.size-1)]
        H_KE_diag2 = [-2*np.ones(r.size)/r**2,np.ones(r.size-1),np.ones(r.size-1)]
        H_KE = -0.5*((2/dr)*diags(H_KE_diag1,[0,1]) + (1/dr**2)*diags(H_KE_diag2,[0,1,-1]))
        
        H_LHY_diag = LHY_coef*abs(phi)**3
        H_LHY = diags(H_LHY_diag,0).toarray()
        	
        H_int_diag = int_coef*abs(phi)**2
        H_int = diags(H_int_diag,0).toarray()
        
        H_trap_diag = V
        H_trap = diags(H_trap_diag,0).toarray()
    
        H_tot = H_KE + H_trap + H_int + H_LHY

        k1 = -dt*np.matmul(H_tot,phi)
        
        # k2 CALCULATION
        H_LHY_diag = LHY_coef*abs(phi + k1/2)**3
        H_LHY = diags(H_LHY_diag,0).toarray()
  
        H_int_diag = int_coef*abs(phi + k1/2)**2
        H_int = diags(H_int_diag,0).toarray
 
        H_tot = H_KE + H_trap + H_int + H_LHY

        k2 = -dt*np.matmul(H_tot,(phi + k1/2))
        
        # k3 CALCULATION
        H_LHY_diag = LHY_coef*abs(phi + k2/2)**3
        H_LHY = diags(H_LHY_diag,0).toarray()
  
        H_int_diag = int_coef*abs(phi + k2/2)**2
        H_int = diags(H_int_diag,[0]).toarray()
 
        H_tot = H_KE + H_trap + H_int + H_LHY
 
        k3 = -dt*np.matmul(H_tot,(phi + k2/2))
       
        # k4 CALCULATION	 
        H_LHY_diag = LHY_coef*abs(phi + k3)**3
        H_LHY = diags(H_LHY_diag,0).toarray()
  
        H_int_diag = int_coef*abs(phi + k3)**2
        H_int = diags(H_int_diag,0).toarray()
 
        H_tot = H_KE + H_trap + H_int + H_LHY
 
        k4 = -dt*np.matmul(H_tot,(phi + k3))
        
        # FINAL RUNGE-KUTTA STEP    
        phi = phi + (1./6)*(k1 + 2*k2 + 2*k3 + k4)
        
        phi[0] = phi[2]
        phi[1] = phi[3]
        phi[-1] = phi[-5]
        phi[-2] = phi[-4]

        # WAVEFUNCTION NORMALISED
        Norm = 4*pi*np.trapz(r**2*abs(phi)**2)*dr
        phi = phi/np.sqrt(Norm) 
        
        # MU AND TOLERANCE CALCULATION
        mu_old = mu
        phi_r = np.gradient(phi,dr)
        mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2 + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
        tol = abs((mu-mu_old)/mu_old)
        count = count + 1
    return(phi,mu)
    
