Directory containing the codes constructed to solve the GP equation derived in Petrov2015. This includes:

~ FILES:

-> petrov_im_tim_rk4.py (A Python script which contains two functions that use the imaginary time method to isolate groundstates of the Petrov GP equation. The differences between the functions lie in the computation of the Kinetic Energy terms which are done: 1) via a for loop; 2) via a matrix multiplication. The quantity of imaginary time steps is an input of these functions so that a large enough number of steps can be used to tend the tolerance of the groundstate towards zero. As this equation is spherically symmetric and the self-bound droplets have a bulk density in their centre, the solver imposes Neumann Boundary Conditions.)

-> petrov_real_tim_rk4.py (A Python script which contains a real time function to propagate the groundstate of the Petrov GPE in time. This includes an observable used to observe the breathing mode from Petrov 2015.)

-> freq_funcs.py (A Python script which contains a short curve fitting function to approximate the frequency of the mode from the real time function. This set of functions will be paired with the imaginary and real time functions in mu_petrov.py to build a parallelised code iterating across N and calculating \mu and \omega_0.)

-> suppress_warns.sh (A Bash script which can be ran by the command: source suppress_warns.sh. This file contains commands which suppress warning messages that arise when running MPI for the parallelised code.)

~ SUBDIRECTORIES:

-> figures (A directory of various different plots generated for testing code and for demonstrating in meetings. Similarly this directory may contain some data files e.g. .csv, .dat files used for generating plots e.g. using pgfplots.)

-> old_rk4 (A directory of outdated Runge-Kutta 4th order codes which had many faults when I was still reminding myself of this method.)

-> euler_fin_dif (A directory of outdated Euler method codes which were the first attemps of solving the Petrov GP equation though again they were riddled with errors. These codes may be revisited however to cement the correct solutions and to contrast with a more inaccurate method.)
