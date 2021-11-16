import os
import numpy as np

#-----------------------------------------------------------------------
# Dimensionless numbers (user-defined)
#-----------------------------------------------------------------------
Ek = 2e-6
Pr = 1.0
Pm = 1.0
Els = 1.0

#-----------------------------------------------------------------------
# Physical parameters (REQUIRED BY SINGE)
#-----------------------------------------------------------------------
# Omega0 : Fluid angular velocity,
# nu     : kinematic viscosity,
# kappa  : diffusivity 1 (thermal),
# kappac : diffusivity 2 (compositional),
# eta    : magnetic diffusivity,
# Ra  : initial guess for the critical Rayleigh number 1 (usually temperature). Ra2 is kept fixed by default.
# Rac : second Rayleigh number (usually compositional). This is a scalar (eigenmodes) or an iterable (for convection).
# BEWARE1: Specify Ra  = 0 (and kappa  = 0) to disable the temperature equation.
# BEWARE2: Specify Rac = 0 (and kappac = 0) to disable the composition equation.
#-----------------------------------------------------------------------
Omega0 = 0.5/Ek
nu = 1
kappa = 1/Pr
kappac = 0
eta = 1/Pm

# Rayleigh numbers
Ra = (0.5*(1+1/Pr)*Ek)**(-4/3)/Pr
Rac = 0

#-----------------------------------------------------------------------
# Imposed radial fields (function of radius only)
# BEWARE: The three functions are required (unless Ra=0, Rac=0)
#-----------------------------------------------------------------------
# N2r1 : radial gradient of the background profile in scalar equation 1 (usually temperature),
# N2r2 : radial gradient of the background profile in scalar equation 2 (usually composition),
# gr   : radial gravity field.
#-----------------------------------------------------------------------
def N2r(r):
	return -r

def N2cr(r):
	return 0*r

def gravity(r):
	"""
	Radial component of the gravity field. The total gravity is of the form -gravity(r)*1_r.
	* Uniform spherical density: gravity(r) = r,
	* Centrally condensed mass: gravity(r) = (r0/r)**2
	"""
	return r

#-----------------------------------------------------------------------
# Background magnetic field
#-----------------------------------------------------------------------
# Btype : 'uniform', 'dipole' or None,
# B0    : magnitude of imposed magnetic field,
# BEWARE: Specify Btype = None to disable the magnetic equation.
#-----------------------------------------------------------------------
Btype = 'uniform'
B0 = np.sqrt(2*Omega0*eta*Els)

#-----------------------------------------------------------------------
# Spectral parameters
#-----------------------------------------------------------------------
# m    : azimuthal wave number. 1D iterable (range, list, ...) for convection or a single value for eigenmodes.
# Lmax : maximum spherical harmonic degree. Parity set up automatically according to 
#		* m even => Lmax odd
#		* m odd  => Lmax even
#		* m = 0  => Lmax even
# sym  : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric. 	
#-----------------------------------------------------------------------
m = [24]
Lmax = 159
sym  = 'pos'

#-----------------------------------------------------------------------
# Radial grid
#-----------------------------------------------------------------------
# r0    : first radial point,
# rf    : last radial point,
# reg   : 'reg' [regular grid] or 'irreg' [irregular grid],
# N     : number of radial intervals (number of points - 1),
# nin   : number of points for the refinement of the inner boundary layer,
# nout  : number of points for the refinement of the outer boundary layer.
#-----------------------------------------------------------------------
r0, rf = 0.4, 1
reg = 'irreg'
N = 300
if reg == 'irreg':
	nin, nout = 15, 15

#-----------------------------------------------------------------------
# Boundary condtions (BC)
#-----------------------------------------------------------------------
# bc_i      : 0 [full sphere], 1 [IC, stress-free], 2 [IC, no-slip],
# bc_o      : 1 [OC, stress-free], 2 [OC, no-slip],
# bc_i_temp : 0 [full sphere], 1 [IC, null flux], 2 [IC, null temperature],
# bc_o_temp : 1 [OC, null flux], 2 [OC, constant temperature],
# bc_i_chi  : 0 [full sphere], 1 [IC, null flux], 2 [IC, null composition],
# bc_o_chi  : 1 [OC, null flux], 2 [OC, null composition].
#-----------------------------------------------------------------------
# Inner boundary
if r0 == 0:
	bc_i, bc_i_temp, bc_i_chi = 0, 0, 0
else:	# TO MODIFY BY THE USER
	bc_i, bc_i_temp, bc_i_chi = 2, 1, 2

# Outer boundary
bc_o, bc_o_temp, bc_o_chi = 2, 1, 2

#-----------------------------------------------------------------------
# Parameters for PETSc/SLEPc
#-----------------------------------------------------------------------
# aliasPY  : python version as called from the terminal (>2.7)
# path_mpi : path to the directory of mpiexec/mpirun. For the mpi version of petsc, use os.environ['PETSC_DIR'] + '/lib/petsc/bin/petscmpiexec' 
# nthreads : nb of mpi process when using petsc/slepc. Default is 1 (mpi does not work well for the convection optimisation).
# inline_slepc : commandline parameters to initialise slepc
# nev : desired number of eigenvalues (> 1 in practice)
# ncv : number of vectors in the Krylov method (ncv > 10 nev)
# eig : target method. Possible choices are
#		'-eps_largest_magnitude'  : SLEPc.EPS.Which.LARGEST_MAGNITUDE
#		'-eps_smallest_magnitude' : SLEPc.EPS.Which.SMALLEST_MAGNITUDE
#		'-eps_largest_real'       : SLEPc.EPS.Which.LARGEST_REAL
#		'-eps_smallest_real'      : SLEPc.EPS.Which.SMALLEST_REAL
#		'-eps_largest_imaginary'  : SLEPc.EPS.Which.LARGEST_IMAGINARY
#		'-eps_smallest_imaginary' : SLEPc.EPS.Which.SMALLEST_IMAGINARY
#		'-eps_target_magnitude'   : SLEPc.EPS.Which.TARGET_MAGNITUDE
#		'-eps_target_real'        : SLEPc.EPS.Which.TARGET_REAL
#		'-eps_target_imaginary'   : SLEPc.EPS.Which.TARGET_IMAGINARY
# tau  : eigenvalue target (complex number). Must be of the form ['-eps_target', '1.0', '2.0'] for e.g. tau= 1+2*1j
# tol  : tolerance of the eigenvalue solver. Default is 1e-12.
# maxit: maximum iterations to converge to an eigenvector. Default is 100.
#-----------------------------------------------------------------------
nthreads = 8
#path_mpi = os.environ['PETSC_DIR'] + '/lib/petsc/bin/petscmpiexec'
path_mpi = 'mpiexec'

inline_slepc = '-eps_balance oneside -eps_conv_abs -st_type sinvert -st_pc_factor_mat_solver_type superlu_dist'
nev   = 5
ncv   = 50
eig   = '-eps_largest_real'
tau   = ['-eps_target', '0', '0']
tol   = 1e-13
maxit = 100

#-----------------------------------------------------------------------
# Parameters for MATLAB (for debugging)
#-----------------------------------------------------------------------
# path_matlab : path to the directory of MATLAB.
# BEWARE: Define path_matlab to use MATLAB instead of PETSc/SLEPc.
#-----------------------------------------------------------------------
path_matlab = '/home/node2/TianyiLi/softwares/matlab-R2021b/bin/matlab'