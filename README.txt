Directory containing the codes constructed to solve the GP equation derived in Petrov2015. This includes:

~ FILES:

-> petrov_im_tim_rk4.py (A Python script which contains two functions that use the imaginary time method to isolate groundstates of the Petrov GP equation. The differences between the functions lie in the computation of the Kinetic Energy terms which are done: 1) via a for loop; 2) via a matrix multiplication. The quantity of imaginary time steps is an input of these functions so that a large enough number of steps can be used to tend the tolerance of the groundstate towards zero. As this equation is spherically symmetric and the self-bound droplets have a bulk density in their centre, the solver imposes Neumann Boundary Conditions.)

-> petrov_im_tim_mats.py (A Python script which attempted to calculate subsequent time steps by constructing the Hamiltonian as a matrix operator to act upon the wavefunction vector. This script has never worked correctly but could be revisited.)

-> petrov_run_plot.py (A Python script which is a bit of a testing-bed for the cleaner functions. This script can call the above functions to propagate groundstates in real time. However, it also contains up-to-date copies of the imaginary time code to test any changes that are made to the imaginary time functions. This has also been used to test the initial breathing mode results.)

-> redone_petrov.py (A Python script which contains the CORRECT imaginary time and real time functions which have been updated since the petrov_im_tim_rk4.py and petrov_run_plot.py codes. This is because there was some confusion previously about avoided imposing some kind of periodic boundary conditions when calculating the kinetic energy terms. Any calculations of imaginary and real time should be undertaken USING THIS SCRIPT.)

-> petrov_real_tim_rk4.py (A Python script which contains a real time function to propagate the groundstate of the Petrov GPE in time. This includes an observable used to observe the breathing mode from Petrov 2015.)

-> freq_funcs.py (A Python script which contains a short curve fitting function to approximate the frequency of the mode from the real time function. This set of functions will be paired with the imaginary and real time functions in mu_petrov.py to build a parallelised code iterating across N and calculating \mu and \omega_0.)

~ SUBDIRECTORIES:

-> figures (A directory of various different plots generated for testing code and for demonstrating in meetings. Similarly this directory may contain some data files e.g. .csv, .dat files used for generating plots e.g. using pgfplots.)

-> old_rk4 (A directory of outdated Runge-Kutta 4th order codes which had many faults when I was still reminding myself of this method.)

-> euler_fin_dif (A directory of outdated Euler method codes which were the first attemps of solving the Petrov GP equation though again they were riddled with errors. These codes may be revisited however to cement the correct solutions and to contrast with a more inaccurate method.)
