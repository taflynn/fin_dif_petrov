import numpy as np

import scipy.integrate as integrate

pi = np.math.pi

def petrov_im_tim_euler(phi,r,dr,dt,N,V):    
    """This functions uses a finite difference methods
    and an Euler time step within an imaginary time setup
    to iteratively isolate the groundstate relating to the
    D.S. Petrov (2015) GPE. This functions requires an 
    initial guess phi_0 (often just a Gaussian centred at
    r = 0) and the relative atom number, N."""
    # COEFFICIENTS
    int_coef= -3*N
    LHY_coef = (5/2)*N**(3/2)
    # TOLERANCE AND MU CALCULATION
    tol = 1
    phi_r = np.gradient(phi,dr)
    mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2 + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
    # HAMILTONIAN TERMS INITIALISING
    H_KE = np.zeros(phi.size)
    H_trap = np.zeros(phi.size)
    H_LHY = np.zeros(phi.size)
    H_int = np.zeros(phi.size)
    KE = np.zeros(phi.size)
    
    count = 0
    
    # BEGIN THE ITERATIVE IMAGINARY TIME LOOP
    while tol>1e-10:
        # CALCULATING THE DERVATIVES FOR THE KINETIC ENERGY TERM
        for j in range(1,phi.size-1):
            KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2    
        # SETTING EACH TERM IN THE HAMILTONIAN
        H_KE = -0.5*KE
        H_trap = V*phi
        H_LHY = LHY_coef*abs(phi)**3*phi
        H_int = int_coef*abs(phi)**2*phi
        # COMPLETING THE EULER TIME STEP
        phi = phi - dt*(H_KE + H_trap + H_LHY + H_int)
        # NEUMANN BOUNDARY CONDITIONS
        phi[0] = phi[2]
        phi[-1] = phi[-3]
        # WAVEFUNCTION NORMALISED
        Norm = 4*pi*np.trapz(r**2*abs(phi)**2)*dr
        phi = phi/np.sqrt(Norm) 
        # MU AND TOLERANCE CALCULATION
        mu_old = mu
        phi_r = np.gradient(phi,dr)
        mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2 + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
        tol = abs((mu-mu_old)/mu_old)
        count = count + 1
        if count>200000:
            count = 'system not converging'
            break
    return(phi,mu,count)

def xi_tau_finder(a_11,a_22,a_12,m_1,m_2):
   
    # PARAMETERS
    g_11 = 4*pi*a_11/m_1
    g_22 = 4*pi*a_22/m_2
    g_12 = 2*pi*a_12*m_1*m_2/(m_1+m_2)
    dg = g_12 + np.sqrt(g_11*g_22)
   
    # MINARDI'S CONSTANTS
    z = m_2/m_1
    u = 1
    x = np.sqrt(g_22/g_11)

    if m_1 == m_2:
        n_01 = (25*pi/1024)*(a_12 + np.sqrt(a_11*a_22))**2/(a_11*a_22*np.sqrt(a_11)*(np.sqrt(a_11) + np.sqrt(a_22))**5)
    else:
        # LAMBDA DEFINE F
        F_int = integrate.quad(lambda k: k**2*(np.sqrt(0.5*(k**2*(1 + (x/z)) + 
                0.25*k**4*(1 +(1/z**2))) + np.sqrt(0.25*((k**2 + 0.25*k**4) - 
                ((x*k**2)/z + (k**4)/(4*z**2)))**2 + u*x*k**4/z)) +
                np.sqrt(0.5*(k**2*(1+(x/z)) + 0.25*k**4*(1+(1/z**2))) -
                np.sqrt(0.25*((k**2 + 0.25*k**4) - ((x*k**2)/z + 
                (k**4)/(4*z**2)))**2 + u*x*k**4/z)) - (1+z)*k**2/(2*z) -
                (1+x) + (1/k**2)*(1 + x**2*z + 4*u*x*z/(1+z))),0,np.inf)
        # CALCULATION OF f
        f = (15/32)*F_int
        # CALCULATION OF n_01
        n_01 = (25*pi/1024)*(1/f**2)*(1/a_11**3)*(dg**2/(g_11*g_22))

    # XI CALCULATION
    xi = np.sqrt(1.5*(np.sqrt(g_22)/m_1 + np.sqrt(g_11)/m_2)/(abs(dg)*np.sqrt(g_11)*n_01))
    
    # TAU CALCULATION
    tau = 1.5*(np.sqrt(g_11) + np.sqrt(g_22))/(abs(dg)*np.sqrt(g_11)*n_01)
    
    return(xi,tau,dg)
