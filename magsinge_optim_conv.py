# General packages
import os
import timeit

# Scientific packages
import numpy as np
import scipy.sparse as sc_sparse
import scipy.io as IO
from scipy.optimize import brentq
from mpi4py import MPI

# Hand-made packages
import sub_op_irreg as sub_op
import sub_eig_irreg as sub_eig
import params

print("""\n########################################################
\tWelcome in MAGSINGE: LINEAR ONSET
####################################################
""")

#-------------------------------------------------------------------#
#							Initialization							#
#-------------------------------------------------------------------#
# Path
CWD = os.getcwd()
abs_path_magsinge = os.path.dirname(os.path.realpath(__file__))
path_mpi = params.path_mpi

# MPI PETSc
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

# Optimization
ra_cache = {}	# cache for optimize_Ra
dlogRa = 1		# initial step to find sign change in growth rate
tol = 1e-6
maxiter = 50
signRa = 1		# assume optimizing positive Rayleigh

#-------------------------------------------------------------------#
#						function to optimize						#
#-------------------------------------------------------------------#
### returns the growth rate for given (Ra, Rac)
def optimize_Ra(Ra, A_withoutRa, nb_l, grid, m, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi, Ra2, gr, flag_temp):
	if (Ra, Ra2) in ra_cache:
		return ra_cache[(Ra, Ra2)]		# don't need to recompute the same value !!
	else:
		# Buoyancy force
		A = sc_sparse.dok_matrix((nb_l, nb_l), dtype=complex)
		A = sub_eig.buoyancy_eig_A(A, grid, m, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi, signRa * 10**Ra, Ra2, gr, flag_temp)
		A = sc_sparse.csr_matrix(A)
		A = A_withoutRa + A
		IO.mmwrite('A.mtx', A)

		print('\t\tx Ra1 = %g, Ra2 = %g' % (signRa * 10**Ra, Ra2))

		# Eigensolver call
		try:
			os.system('cp '+ abs_path_magsinge + '/magsinge_eig.m .')
			os.system('cp '+ abs_path_magsinge + '/mmread.m .')
			runcmd = params.path_matlab + ' -batch' + ' "magsinge_eig"'
			start_time = timeit.default_timer()
			os.system(runcmd)
			np.save('Real_Eigenvec.npy', np.loadtxt('Real_Eigenvec.txt'))
			np.save('Imag_Eigenvec.npy', np.loadtxt('Imag_Eigenvec.txt'))
			os.system('rm Real_Eigenvec.txt')
			os.system('rm Imag_Eigenvec.txt')
			elapsed = timeit.default_timer() - start_time
			os.system('rm magsinge_eig.m mmread.m')
		except:
			runcmd = 'python3 ' + abs_path_magsinge + '/magsinge_petsc_mpi.py ' + params.inline_slepc
			if params.nthreads > 1:
				runcmd = path_mpi + ' -n ' + str(params.nthreads) + ' ' + runcmd
			start_time = timeit.default_timer()
			os.system(runcmd)
			elapsed = timeit.default_timer() - start_time

		# Optim criterion
		eigval = np.loadtxt('Eigenval.txt')
		indmax = np.argmax(eigval[:, 0])		# take the eigenvalue with the largest real
		sigma_c, omega_c = eigval[indmax, 0], eigval[indmax, 1]
		print('\t\t\t => Growth rate = %g (elapsed %.4g s)' % (sigma_c, elapsed))
		ra_cache[(Ra, Ra2)] = sigma_c

		optimarr = np.array([signRa * 10**Ra, Ra2, sigma_c, omega_c])
		if sigma_c >= 0.0:
			if os.path.isfile('optimpos_iter.out') == True:
				optimarr = np.vstack([np.loadtxt('optimpos_iter.out'), optimarr])
			np.savetxt('optimpos_iter.out', optimarr)
		else:
			if os.path.isfile('optimneg_iter.out') == True:
				optimarr = np.vstack([np.loadtxt('optimneg_iter.out'), optimarr])
			np.savetxt('optimneg_iter.out', optimarr)

		return sigma_c

### assumes f is an increasing function of x.
def bracket_brentq(f, x1, x2=None, dx=0.3, tol=1e-6, maxiter=200, args=None):
	y1 = f(x1, *args)
	dx = abs(dx)					# dx must be positive.
	if x2 is not None:				# check that we actually have a bracket
		y2 = f(x2, *args)
		if y2*y1 >0:				# we don't have a bracket !!
			if (abs(y2) < abs(y1)):
				x1, y1 = x2, y2		# start from the value closest to the root
			x = None				# search needed.
	if x2 is None:					# search for a bracket
		x2 = x1
		if y1 > 0: dx = -dx			# up or down ?
		while x2 > 1:
			x2 += dx
			y2 = f(x2, *args)
			if y2*y1 < 0: break
			x1, y1 = x2, y2
	if x2 <= 1.:
		return 0.0
	# Now that we know that the root is between x1 and x2, we use Brent's method:
	x0 = brentq(f, x1, x2, maxiter=maxiter, xtol=tol, rtol=tol, args=args)
	return x0

#-------------------------------------------------------------------#
#							Radial grid								#
#-------------------------------------------------------------------#
if params.reg == 'reg':
	grid = sub_op.radial_grid(params.r0, params.rf, params.N)
	grid.mesh_reg()
elif params.reg == 'irreg':
	grid = sub_op.radial_grid(params.r0, params.rf, params.N)
	grid.mesh_irreg(params.nin, params.nout)

if mpi_rank == 0:
	print('o Radial grid')
	if params.reg == 'reg':
		print('\t* h = ' + str(grid.h))
	elif params.reg == 'irreg':
		print('\t* N = ' + str(grid.N))
		print('\t* hmin = ' + str(grid.dmin))
		print('\t* hmax = '+ str(grid.dmax))

comm.Barrier()

#-------------------------------------------------------------------#
#							Radial fields							#
#-------------------------------------------------------------------#
N2 = params.N2r(grid.r)
gr = params.gravity(grid.r)

if np.any(params.Rac):				# Can be single value or a list
	flag_temp = 2
	N2c = params.N2cr(grid.r)
	bc_i_chi, bc_o_chi = params.bc_i_chi, params.bc_o_chi
else:
	flag_temp = 1
	N2c = np.zeros_like(grid.r)
	bc_i_chi, bc_o_chi = 1, 1		# Default value. Does not play any role because composition is disabled!

flag_mag = 0						# Default value.
if params.Btype == 'uniform' or params.Btype == 'dipole':
	flag_mag = 1
	Bfield = sub_op.Bfield(params.Btype, params.B0, grid)

### parameter space (m, Rac) ###
mc_all = np.array(params.m)
Rac_all = np.ones((1, np.size(mc_all))) * np.array(params.Rac).reshape((-1, 1))
mc_all = mc_all.reshape((1, -1)) * np.ones((np.size(params.Rac), 1))

par_idx0 = mpi_rank * np.size(mc_all) // mpi_size
par_idx1 = (mpi_rank+1) * np.size(mc_all) // mpi_size
comm.Barrier()
print("\no Process %d optimizing %d parameter sets (out of %d)" % (mpi_rank, par_idx1-par_idx0, np.size(mc_all)))
comm.Barrier()

#-------------------------------------------------------------------#
#					Parallel loop over m and for Ra					#
#-------------------------------------------------------------------#
# Allocation on each process
mc_vec = mc_all.flat[par_idx0:par_idx1]				# m
Rac_vec = np.zeros(mc_vec.shape, dtype=float)		# First Ra to optimize (thermal)
Rac2_vec = Rac_all.flat[par_idx0:par_idx1]			# Second Ra (imposed)
omegac_vec = np.zeros(mc_vec.shape, dtype=float)	# frequency (to fine)
sigmac_vec = np.zeros(mc_vec.shape, dtype=float)	# growth rate (to find)
count = 0

comm.Barrier()

# Do previous results exist ?
try:
	result_prev = np.loadtxt('m_Rac_omega_sigma_%d.txt' % mpi_rank)
	if (result_prev.shape == (mc_vec.size, 5)) and (np.all(mc_vec == result_prev[:, 0])) and (np.allclose(Rac2_vec, result_prev[:,2])):
		Rac_vec = result_prev[:,1]
		omegac_vec = result_prev[:,3]
		sigmac_vec = result_prev[:,4]
		print('o Continuing previous job.')
	else:
		print('o Job has changed. Starting from scratch.')
except:
	pass

# Start value for finding the critical Rayleigh (in log-space)
signRa = np.sign(params.Ra)
Ra_crit = np.log10(np.abs(params.Ra))

for m, Rac in zip(mc_vec, Rac2_vec):
	if (Rac_vec[count] == 0.0) and (omegac_vec[count] == 0.0) and (sigmac_vec[count] == 0.0):		# compute only if not already computed
		m = int(m)
		print('\no Onset for fixed m = %d, Ra2 = %g' % (m, Rac))

		#----------------
		# Init
		#----------------
		path_mdir = 'm' + str(m)
		os.chdir(path_mdir)
		
		#----------------
		# Spectral scalar variables
		#----------------
		Lmax, sym, sh = sub_eig.set_shtns(params.sym, params.Lmax, m)
		pol, tor, temp, chi, polb, torb = sub_eig.nb_eig_vec(m, params.N, Lmax, sym, params.bc_i, params.bc_o, params.bc_i_temp, params.bc_o_temp, bc_i_chi, bc_o_chi)

		#----------------
		# Size of the matrices A and B
		#----------------
		if flag_temp == 1:
			nb_l = pol.nbel_vec + tor.nbel_vec + temp.nbel_vec
		elif flag_temp == 2:
			nb_l = pol.nbel_vec + tor.nbel_vec + temp.nbel_vec + chi.nbel_vec

		if flag_mag == 1:
			nb_l += polb.nbel_vec + torb.nbel_vec

		#----------------
		# Eigenvalue optimization
		#----------------
		# Reading of A
		A_withoutRa = IO.mmread('A_withoutRa.mtx')
		A_withoutRa = sc_sparse.csr_matrix(A_withoutRa)

		# Find the zero from initial guess Ra_crit
		print('\t* Optimization on Ra1')
		ra_cache = {}		# clear cache for optimize_Ra
		optargs = (A_withoutRa, nb_l, grid, m, Lmax, sym, params.bc_i, params.bc_o, params.bc_i_temp, params.bc_o_temp, bc_i_chi, bc_o_chi, Rac, gr, flag_temp)
		Ra_crit = bracket_brentq(optimize_Ra, Ra_crit, dx=dlogRa, tol=tol, maxiter=maxiter, args=optargs)
		dlogRa = 0.2		# for subsequent searches, we should use a lower value for dlogRa.

		# Storage and cleaning
		Rac_vec[count] = signRa * 10**Ra_crit

		# Onset marginal mode for each m
		print('\n\t* Eigenmode at the linear onset')
		os.system('rm Eigenval.txt')

		# Buoyancy force
		A = sc_sparse.dok_matrix((nb_l, nb_l), dtype=complex)
		A = sub_eig.buoyancy_eig_A(A, grid, m, Lmax, sym, params.bc_i, params.bc_o, params.bc_i_temp, params.bc_o_temp, bc_i_chi, bc_o_chi, Rac_vec[count], Rac, gr, flag_temp)
		A = sc_sparse.csr_matrix(A)
		A = A + A_withoutRa
		IO.mmwrite('A.mtx', A)
		del A

		try:
			os.system('cp '+ abs_path_magsinge + '/magsinge_eig.m .')
			os.system('cp '+ abs_path_magsinge + '/mmread.m .')
			runcmd = params.path_matlab + ' -batch' + ' "magsinge_eig"'
			runcmd = params.path_matlab + ' -batch' + ' "magsinge_eig"'
			os.system(runcmd)
			np.save('Real_Eigenvec.npy', np.loadtxt('Real_Eigenvec.txt'))
			np.save('Imag_Eigenvec.npy', np.loadtxt('Imag_Eigenvec.txt'))
			os.system('rm Real_Eigenvec.txt')
			os.system('rm Imag_Eigenvec.txt')
			os.system('rm magsinge_eig.m mmread.m')
		except:
			runcmd = 'python3 ' + abs_path_magsinge + '/magsinge_petsc_mpi.py ' + params.inline_slepc
			if params.nthreads > 1:
				runcmd = path_mpi + ' -n ' + str(params.nthreads) + ' ' + runcmd
			os.system(runcmd)

		# Load result of eigen solve:
		eigval = np.loadtxt('Eigenval.txt')
		indmin = np.abs(eigval[:, 0]).argmin()
		omegac_vec[count] = eigval[indmin, 1]
		sigmac_vec[count] = eigval[indmin, 0]
		os.rename('Real_Eigenvec.npy', 'Real_Eigenvec_Rac%.3g.npy' % Rac)	# keep mode for every Rac
		os.rename('Imag_Eigenvec.npy', 'Imag_Eigenvec_Rac%.3g.npy' % Rac)

		# Save result and print message
		os.chdir(CWD)
		result_all = np.vstack((mc_vec, Rac_vec, Rac2_vec, omegac_vec, sigmac_vec)).transpose()
		np.savetxt('m_Rac_omega_sigma_%d.txt' % mpi_rank, result_all)		# overwrite results
		print("\t\t=> Critical Rayleigh number for m = " + str(m) + " is Ra_crit = %.3e" % (signRa * 10**Ra_crit))
		print("\t\t=> Frequency at onset for m = " + str(m) + " is omega = %.3e" % (omegac_vec[count]))

	count += 1

comm.Barrier()

#-------------------------------------------------------------------#
#					Gathering and save on process 0					#
#-------------------------------------------------------------------#
if mpi_rank == 0:
	result_all = np.loadtxt('m_Rac_omega_sigma_0.txt')
	for indmpi in range(1, mpi_size):
		resulti = np.loadtxt('m_Rac_omega_sigma_%d.txt' % indmpi)
		result_all = np.vstack((result_all, resulti))
	np.savetxt('m_Rac_omega_sigma.txt', result_all)		# overwrite results
	os.system('rm m_Rac_omega_sigma_*.txt')
	print("\nFinished. You may remove the matrices with 'rm m*/*.mtx'")