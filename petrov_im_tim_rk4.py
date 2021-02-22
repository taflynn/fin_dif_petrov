# PACKAGES
import numpy as np
# CONSTANTS 
pi = np.math.pi

###############################################################################
# RK4 IMAGINARY TIME FUNCTION (W/ FOR LOOPS IN KE TERM)
def petrov_im_tim_rk4(phi,r,dr,dt,N,V,int_gas,im_t_steps):
	# GPE COEFFICIENTS
	if int_gas == 0:
		int_coef = 0
		LHY_coef = 0
	elif int_gas == 1:    
		int_coef = -3*N
		LHY_coef = (5/2)*N**(3/2)    

	# INITIAL MU
	phi_r = np.gradient(phi,dr) # phi derivative
	mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2  + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
	tol = 1 # initialise tolerance
	count = 1 # initialise counter for imaginary time

	# INITIALISE HAMILTONIAN ARRAYS
	H_KE = np.zeros(phi.size)
	H_LHY = np.zeros(phi.size)
	H_int = np.zeros(phi.size)
	H_trap = np.zeros(phi.size)
	KE = np.zeros(phi.size)

	# IMAGINARY TIME WHILE LOOP
	for i in range(0,im_t_steps+1):
		# k1 CALCULATION
		for j in range(1,phi.size-1):
			KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2  
		# HAMILTONIAN TERMS    
		H_KE = -0.5*KE # KE term
		H_LHY = LHY_coef*abs(phi)**3*phi # LHY term
		H_int = int_coef*abs(phi)**2*phi # s-wave term
		H_trap = V*phi # potential term

		k1 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*phi)

		# k2 CALCULATION
		for j in range(1,phi.size-1):
			KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2 + 0.5*(2/r[j])*(k1[j+1] - k1[j])/dr + 0.5*(k1[j+1] - 2*k1[j] + k1[j-1])/dr**2  
		# HAMILTONIAN TERMS
		H_KE = -0.5*KE # KE term
		H_LHY = LHY_coef*abs(phi + k1/2)**3*(phi + k1/2) # LHY term
		H_int = int_coef*abs(phi + k1/2)**2*(phi + k1/2) # s-wave term
		H_trap = V*(phi + k1/2) # potential term

		k2 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*(phi + k1/2))

		# k3 CALCULATION
		for j in range(1,phi.size-1):
			KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2 + 0.5*(2/r[j])*(k2[j+1] -k2[j])/dr + 0.5*(k2[j+1] - 2*k2[j] + k2[j-1])/dr**2  
		# HAMILTONIAN TERMS
		H_KE = -0.5*KE
		H_LHY = LHY_coef*abs(phi + k2/2)**3*(phi + k2/2)
		H_int = int_coef*abs(phi + k2/2)**2*(phi + k2/2)
		H_trap = V*(phi + k2/2)

		k3 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*(phi + k2/2))

		#k4 CALCULATION
		for j in range(1,phi.size-1):
			KE[j] = (2/r[j])*(phi[j+1] - phi[j])/dr + (phi[j+1] - 2*phi[j] + phi[j-1])/dr**2 + (2/r[j])*(k3[j+1] -k3[j])/dr + (k3[j+1] - 2*k3[j] + k3[j-1])/dr**2  
		# HAMILTONIAN TERMS
		H_KE = -0.5*KE # KE term
		H_LHY = LHY_coef*abs(phi + k3)**3*(phi + k3) # LHY term
		H_int = int_coef*abs(phi + k3)**2*(phi + k3) # s-wave term
		H_trap = V*(phi + k3) # potential term

		k4 = -dt*(H_KE + H_trap + H_LHY + H_int - mu*phi)

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
		# ITERATE COUNTER
		count = count + 1
	return(phi,mu)

###############################################################################
#RK4 IMAGINARY TIME FUNCTION (W/ MATRICES IN KE TERMS)
def petrov_im_tim_rk4_mat(phi,r,dr,dt,N,V,int_gas,im_t_steps):
	# GPE COEFFICIENTS
	if int_gas == 0:
		int_coef = 0
		LHY_coef = 0
	elif int_gas == 1:    
		int_coef = -3*N
		LHY_coef = (5/2)*N**(3/2)    

	# INITIAL MU AND TIME
	t = 0
	phi_r = np.gradient(phi,dr) # derivative of wavefunction
	mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2  + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
	tol_mu = 1 # initialise tolerance
	count = 1 # initialise imaginary time counter
	dens = max(N*abs(phi)**2)

	# DIFFERENTIAL OPERATORS
	Dr = (1/(2*dr))*(-1*np.eye(phi.size-2,phi.size,k=0,dtype=float) + np.eye(phi.size-2,phi.size,k=2,dtype=float))
	Dr2 =  (1/dr**2)*(np.eye(phi.size-2,phi.size,k=0,dtype=float) + -2*np.eye(phi.size-2,phi.size,k=1,dtype=float) + np.eye(phi.size-2,phi.size,k=2,dtype=float))

	# INITIALISING ARRAYS
	H_KE = np.zeros(phi.size)
	H_LHY = np.zeros(phi.size)
	H_int = np.zeros(phi.size)
	H_trap = np.zeros(phi.size)
	KE = np.zeros(phi.size)
	k1 = np.zeros(phi.size)
	k2 = np.zeros(phi.size)
	k3 = np.zeros(phi.size)
	k4 = np.zeros(phi.size)
	
	# IMAGINARY TIME WHILE LOOP
	for l in range(0,im_t_steps+1):
		# k1 CALCULATION
		KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi
		# HAMILTONIAN TERMS
		H_KE[1:-1] = -0.5*KE[1:-1] # KE term
		H_LHY[1:-1] = LHY_coef*abs(phi[1:-1])**3*phi[1:-1] # LHY term
		H_int[1:-1] = int_coef*abs(phi[1:-1])**2*phi[1:-1] # s-wave term
		H_trap[1:-1] = V[1:-1]*phi[1:-1] # potential term

		k1[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*phi[1:-1])

		k1[0] = k1[1]
		k1[-1] = k1[-2]

		# k2 CALCULATION
		KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k1) + Dr2 @ k1) 
		# HAMILTONIAN TERMS
		H_KE[1:-1] = -0.5*KE[1:-1] # KE term
		H_LHY[1:-1] = LHY_coef*abs(phi[1:-1] + k1[1:-1]/2)**3*(phi[1:-1] + k1[1:-1]/2) # LHY term
		H_int[1:-1] = int_coef*abs(phi[1:-1] + k1[1:-1]/2)**2*(phi[1:-1] + k1[1:-1]/2) # s-wave term
		H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k1[1:-1]/2) # potential term

		k2[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k1[1:-1]/2))

		k2[0] = k2[1]
		k2[-1] = k2[-2]

		# k3 CALCULATION
		KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + 0.5*((2/r[1:-1])*(Dr @ k2) + Dr2 @ k2)  
		# HAMILTONIAN TERMS 
		H_KE[1:-1] = -0.5*KE[1:-1] # KE term
		H_LHY[1:-1] = LHY_coef*abs(phi[1:-1] + k2[1:-1]/2)**3*(phi[1:-1] + k2[1:-1]/2) # LHY term
		H_int[1:-1] = int_coef*abs(phi[1:-1] + k2[1:-1]/2)**2*(phi[1:-1] + k2[1:-1]/2) # s-wave term
		H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k2[1:-1]/2) # potential term

		k3[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k2[1:-1]/2))

		k3[0] = k3[1]
		k3[-1] = k3[-2]

		#k4 CALCULATION
		KE[1:-1] = (2/r[1:-1])*(Dr @ phi) + Dr2 @ phi + ((2/r[1:-1])*(Dr @ k3) + Dr2 @ k3)  
		# HAMILTONIAN TERMS
		H_KE[1:-1] = -0.5*KE[1:-1] # KE term
		H_LHY[1:-1] = LHY_coef*abs(phi[1:-1] + k3[1:-1])**3*(phi[1:-1] + k3[1:-1]) # LHY term
		H_int[1:-1] = int_coef*abs(phi[1:-1] + k3[1:-1])**2*(phi[1:-1] + k3[1:-1]) # s-wave term
		H_trap[1:-1] = V[1:-1]*(phi[1:-1] + k3[1:-1]) # potential term

		k4[1:-1] = -dt*(H_KE[1:-1] + H_trap[1:-1] + H_LHY[1:-1] + H_int[1:-1] - mu*(phi[1:-1] + k3[1:-1]))

		k4[0] = k4[1]
		k4[-1] = k4[-2]

		# FINAL RUNGE-KUTTA STEP    
		phi[1:-1] = phi[1:-1] + (1./6)*(k1[1:-1] + 2*k2[1:-1] + 2*k3[1:-1] + k4[1:-1])

		# NEUMANN BOUNDARY CONDITIONS
		# phi(j+1) - phi(j) = 0
		phi[0] = phi[1]
		phi[-1] = phi[-2]

		# WAVEFUNCTION NORMALISED
		Norm = 4*pi*np.trapz(r**2*abs(phi)**2)*dr
		phi = phi/np.sqrt(Norm) 

		# ITERATE TIME     
		t = t + dt    

		# ITERATE COUNTER
		count = count + 1

		# MU TOLERANCE CALCULATION
		mu_old = mu
		phi_r = np.gradient(phi,dr)
		mu = np.trapz(r**2*(0.5*abs(phi_r)**2 + V*abs(phi)**2 + int_coef*abs(phi)**4 + LHY_coef*abs(phi)**5))/np.trapz(r**2*abs(phi)**2)
		tol_mu = abs((mu-mu_old)/mu_old)

		# DENSITY MODE TOLERANCE CALCULATION
		dens_old = dens
		dens = max(N*abs(phi)**2)
		tol_mode = abs((dens-dens_old)/dens_old)

		# ITERATE COUNTER
		count = count + 1
	return(phi,mu,tol_mu,tol_mode,t)

