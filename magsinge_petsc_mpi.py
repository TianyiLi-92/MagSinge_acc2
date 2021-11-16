import sys, time

import numpy as np
import scipy.sparse as sc_sparse
import scipy.io as IO

import params

import slepc4py
slepc4py.init(sys.argv + [str(params.eig)] + [str(params.tau[0]), str(params.tau[1]) + '+' + str(params.tau[2]) + 'i'])

from petsc4py import PETSc
from slepc4py import SLEPc

#-------------------------------------------------------------------#
#					Parameters for PETSc / SLEPc					#
#-------------------------------------------------------------------#
rank = PETSc.COMM_WORLD.getRank()
opts = PETSc.Options()

#-------------------------------------------------------------------#
#							Solver SLEPc							#
#-------------------------------------------------------------------#
def solv_eig_syst(A, B, nev, ncv, tol=10**(-8), maxit=100):
	#----------------------------------------------------------------
	""" Call of SLEPc solver for the GEP AX = kBX.
	Inputs :
		* A      : sparse matrix N*N PETSc,
		* B      : sparse matrix N*N PETSc,
		* nev    : nb of eigenvalues to compute at minimum [integer],
		* ncv    : size of the Krylov space to use for the eigenvalue comptuations ncv >> 2 nev for accurate computations,
		* tol    : tolerance [default = 10**(-8)],
		* mpd    : maximum projected dimension [default = 0],
	Output :
		* sol    : class.
	"""
	#----------------------------------------------------------------
	# Initialization
	class solution:
		"Definition of the solution class"
	sol = solution()

	# Solver E
	E = SLEPc.EPS()
	E.create(SLEPc.COMM_WORLD)
	E.setOperators(A, B)
	E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
	E.setDimensions(nev, ncv)
	E.setTolerances(tol, maxit)

	if   params.eig == '-eps_largest_magnitude':
		E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
	elif params.eig == '-eps_smallest_magnitude':
		E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
	elif params.eig == '-eps_largest_real':
		E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
	elif params.eig == '-eps_smallest_real':
		E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
	elif params.eig == '-eps_largest_imaginary':
		E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
	elif params.eig == '-eps_smallest_imaginary':
		E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_IMAGINARY)
	else:
		if   params.eig == '-eps_target_magnitude':
			E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
		elif params.eig == '-eps_target_real':
			E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
		elif params.eig == '-eps_target_imaginary':
			E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_IMAGINARY)
		E.setTarget(float(params.tau[1]) + 1j*float(params.tau[2]))

	# Solving
	E.setFromOptions()
	tic = time.process_time()
	E.solve()
	toc = time.process_time()

	# Getting the outputs
	sol.its                   = E.getIterationNumber()
	sol.neps_type             = E.getType()
	sol.tol, sol.maxit        = E.getTolerances()
	sol.nev, sol.ncv, sol.mpd = E.getDimensions()
	sol.nconv                 = E.getConverged()
	sol.k                     = np.zeros(sol.nconv, dtype=complex)
	sol.error                 = np.zeros(sol.nconv)
	sol.vec                   = np.zeros((sol.nconv, nb_l), dtype=complex)
	sol.tau                   = E.getTarget()
	if sol.nconv > 0:
		# Initialization of the eigenvectors
		vr, wr = A.getVecs()
		vi, wi = A.getVecs()
		for i in range(sol.nconv):
			# Creation of values ​​and pooling on processor 0
			k     = E.getEigenpair(i, vr, vi)
			error = E.computeError(i)
			tozero, VR = PETSc.Scatter.toZero(vr)
			tozero.begin(vr, VR)
			tozero.end(vr, VR)
			tozero.destroy()
			# Assignment
			sol.k[i]     = k
			sol.error[i] = error
			if rank == 0:
				sol.vec[i] = VR
				sol.time   = toc - tic

	return sol

#-------------------------------------------------------------------#
#							PETSc Matrix B							#
#-------------------------------------------------------------------#
# Import of CSR sparse matrix
B = IO.mmread('B.mtx')
B = sc_sparse.csr_matrix(B)
nb_l, nb_c = B.shape
nbl = opts.getInt('nbl', nb_l)

# Fast PETSc conversion
MB = PETSc.Mat()
MB.create(PETSc.COMM_WORLD)
MB.setSizes([nbl, nbl])
MB.setType('mpiaij')
MB.setFromOptions()
MB.setUp()

Istart, Iend = MB.getOwnershipRange()
indptrB = B[Istart:Iend, :].indptr
indicesB = B[Istart:Iend, :].indices
dataB = B[Istart:Iend, :].data

del B

MB.setPreallocationCSR(csr=(indptrB, indicesB))
MB.setValuesCSR(indptrB, indicesB, dataB)

MB.assemblyBegin()
MB.assemblyEnd()

del indptrB, indicesB, dataB

#-------------------------------------------------------------------#
#							PETSc Matrix A							#
#-------------------------------------------------------------------#
# Import of CSR non-zero elements
A = IO.mmread('A.mtx')
A = sc_sparse.csr_matrix(A)

# Fast PETSc conversion
MA = PETSc.Mat()
MA.create(PETSc.COMM_WORLD)
MA.setSizes([nbl, nbl])
MA.setType('mpiaij')
MA.setFromOptions()
MA.setUp()

Istart, Iend = MA.getOwnershipRange()
indptrA = A[Istart:Iend, :].indptr
indicesA = A[Istart:Iend, :].indices
dataA = A[Istart:Iend, :].data

del A

MA.setPreallocationCSR(csr=(indptrA, indicesA))
MA.setValuesCSR(indptrA, indicesA, dataA)

MA.assemblyBegin()
MA.assemblyEnd()

del indptrA, indicesA, dataA

#-------------------------------------------------------------------#
#							Solver call								#
#-------------------------------------------------------------------#
sol = solv_eig_syst(MA, MB, params.nev, params.ncv, maxit=params.maxit, tol=params.tol)

#-------------------------------------------------------------------#
#			Save of eigenmodes and parameters simulations			#
#-------------------------------------------------------------------#
if rank == 0:
	# Eigenvalues
	txt1 = np.vstack([np.real(sol.k), np.imag(sol.k)]).transpose()
	title_txt1 = 'Eigenval.txt'
	np.savetxt(title_txt1, txt1)
	# Eigenvectors
	title_txt21 = 'Real_Eigenvec.npy'
	title_txt22 = 'Imag_Eigenvec.npy'
	np.save(title_txt21, np.real(sol.vec))
	np.save(title_txt22, np.imag(sol.vec))