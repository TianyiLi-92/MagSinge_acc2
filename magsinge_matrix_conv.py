# General packages
import os,shutil
import time

# Scientific packages
import numpy as np
import scipy.sparse as sc_sparse
import scipy.io as IO
from mpi4py import MPI

# Hand-made packages
import sub_op_irreg as sub_op
import sub_eig_irreg as sub_eig
import params

#-------------------------------------------------------------------#
#							Initialization							#
#-------------------------------------------------------------------#
CWD = os.getcwd()
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

if mpi_rank == 0:
	print("""\n########################################################
\tWelcome in MAGSINGE: LINEAR CONVECTION
########################################################
""")

comm.Barrier()

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

#-------------------------------------------------------------------#
#			Creating bulk matrices in parallel (without Ra)			#
#-------------------------------------------------------------------#
mc_all = np.array(params.m)
par_idx0 = mpi_rank * np.size(mc_all) // mpi_size
par_idx1 = (mpi_rank+1) * np.size(mc_all) // mpi_size
mc_vec = mc_all.flat[par_idx0:par_idx1]			# m

if mpi_rank == 0:
	print('\no Building the A and B matrices on ' + str(mpi_size) + ' MPI processes')

comm.Barrier()

for m in mc_vec:
	#----------------
	# Init
	#----------------
	path_mdir = 'm' + str(m)
	if os.path.isdir(path_mdir) == False:
		os.mkdir(path_mdir)
	shutil.copy('params.py', path_mdir)
	os.chdir(path_mdir)

	#----------------
	# Spectral scalar variables: same resolution for each (m, parity)
	#----------------
	Lmax, sym, sh = sub_eig.set_shtns(params.sym, params.Lmax, m.item())
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
	# Eigenvalue matrices AX = lambda B X
	#----------------
	# Filling of B
	tic = time.process_time()
	B = sc_sparse.dok_matrix((nb_l, nb_l), dtype=complex)
	B = sub_eig.hydro_eig_B(B, grid, m, Lmax, sym, params.bc_i, params.bc_o, params.bc_i_temp, params.bc_o_temp, bc_i_chi, bc_o_chi, flag_temp)
	B = sub_eig.temp_eig_B(B, grid, m, Lmax, sym, params.bc_i, params.bc_o, params.bc_i_temp, params.bc_o_temp, bc_i_chi, bc_o_chi, flag_temp)

	if flag_mag == 1:
		B = sub_eig.magnetic_eig_B(B, grid, m, Lmax, sym, params.bc_i, params.bc_o, params.bc_i_temp, params.bc_o_temp, bc_i_chi, bc_o_chi, flag_temp)

	# Filling of A
	A = sc_sparse.dok_matrix((nb_l, nb_l), dtype=complex)
	A = sub_eig.hydro_eig_A(A, grid, sh, m, Lmax, sym, params.bc_i, params.bc_o, params.bc_i_temp, params.bc_o_temp, bc_i_chi, bc_o_chi, params.Omega0, params.nu, flag_temp)
	A = sub_eig.temp_eig_A(A, grid, m, Lmax, sym, params.bc_i, params.bc_o, params.bc_i_temp, params.bc_o_temp, bc_i_chi, bc_o_chi, params.kappa, params.kappac, 0, 0, N2, N2c, gr, flag_temp)

	if flag_mag == 1:
		A = sub_eig.magnetic_eig_A(A, grid, Bfield, sh, m, Lmax, sym, params.bc_i, params.bc_o, params.bc_i_temp, params.bc_o_temp, bc_i_chi, bc_o_chi, params.eta, flag_temp)

	# Save of on the HDD
	A = sc_sparse.csr_matrix(A)
	B = sc_sparse.csr_matrix(B)
	IO.mmwrite('A_withoutRa.mtx', A)
	IO.mmwrite('B.mtx', B)
	toc = time.process_time()
	timeMat = toc - tic

	# Check for errors:
	if ~np.all(np.isfinite(A.data)):
		print('Error in matrix A, inf/nan found')
	if ~np.all(np.isfinite(B.data)):
		print('Error in matrix B, inf/nan found')
	del A, B
	print('\tx m = ' + str(m) + ': ' + str(round(timeMat,2)) + ' s')
	os.chdir(CWD)

comm.Barrier()