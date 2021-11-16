import numpy as np
from scipy.optimize import brentq
import math

#-------------------------------------------------------------------#
#						Radial grid generation						#
#-------------------------------------------------------------------#
class radial_grid():
	def __init__(self, r0, rf, N):
		self.r0 = r0				# initial radius [real]
		self.rf = rf				# final radius [real]
		self.N  = N					# number of intervals [integer]

	# Regular grid
	def mesh_reg(self):
		self.h  = (self.rf - self.r0) / float(self.N)
		self.r  = self.r0 + self.h*np.arange(0, self.N+1, dtype=float)
		self.dr = self.h*np.ones(self.N, dtype=float)

	# The sum of the N first powers of the unknown x is y.
	# we assume x<1
	def decre(self, N, y):
		e = y/(y+1)
		f = N+1
		x = e
		while np.abs(y + 1 - (1-x**f)/(1-x)) > 0.0001:
			x = e + x**f/(y+1)
		return x

	# Generate a grid with densification in Boundary Layers, based on code from D. Jault.
	def mesh_irreg(self, nin=0, nout=0):
		# Initialization
		self.nin  = nin				# nb of points to densify the internal BL
		self.nout = nout			# nb of points to densify the external BL
		nb = self.N - (self.nin + self.nout)
		NG = 0
		NM = self.N
		nr1 = NG + (int(nb/4) + self.nin)
		nr2 = NM - (int(nb/4) + self.nout)

		# Verification
		if(self.nin + self.nout) >= self.N:
			print('nin+nout >= N')
			return
		if self.r0 == 0:
			nr1 = NG + (int(3*nb/20) + self.nin)
		if (nr1 >= nr2) or (nr1 <= NG) or (nr2 >= NM):
			print('Switch to regular grid')
			self.mesh_reg()
			return

		self.r  = np.zeros(self.N+1, dtype=float)
		self.dr = np.zeros(self.N, dtype=float)

		# Irregular grid generation
		# nr1 index of external shell of internal BL
		# nr2 index of internal shell of external BL
		h = (self.rf - self.r0)*0.15
		hu = (self.rf - self.r0 - 2*h)/(nr2 - nr1)
		self.r[NG] = self.r0
		self.r[NM] = self.rf
		self.r[nr1] = self.r0 + h
		self.r[nr2] = self.rf - h
		# Uniform grid in the bulk
		for i in range(nr1+1, nr2):
			self.r[i] = self.r[nr1] + (i - nr1)*hu
		# Outer BL
		q = self.decre(NM-nr2, h/hu)
		e = hu
		for i in range(nr2+1, NM):
			e = e*q
			self.r[i] = self.r[i-1] + e
		# Inner BL
		q = self.decre(nr1-NG, h/hu)
		e = hu
		for i in range(nr1-1, NG, -1):
			e = e*q
			self.r[i] = self.r[i+1] - e
		# Radial path
		for k in range(0, self.N):
			self.dr[k] = self.r[k+1] - self.r[k]
		self.dmin = self.dr.min()
		self.dmax = self.dr.max()

	# Find b from the percentage nb/N.
	# Both internal BL and external BL takes up 15% * (rf - r0).
	def find_b(self, b, p):
		return 1 - np.arctanh( (1-0.15*2)*np.tanh(b) )/b - p

	# Generate a grid with densification in Boundary Layers.
	# r = r0 + (tanh(b*\eta)/tanh(b) + 1) / 2 * (rf - r0), -1 <= \eta <= 1
	def mesh_tanh(self, nb):
		# Initialization
		self.nb = nb				# nb of points in the BL [nin = nout = nb/2]

		# Verification
		p = self.nb/self.N
		if p < self.find_b(b=0.1, p=0):
			print('nb is too small')
			return
		if p > self.find_b(b=10, p=0):
			print('nb is too large')
			return

		b = brentq(self.find_b, 0.1, 10, p)
		eta = np.linspace(-1, 1, self.N+1, dtype=float)
		self.r = self.r0 + (np.tanh(b*eta)/np.tanh(b) + 1) / 2 * (self.rf - self.r0)

		# Radial path
		self.dr = np.zeros(self.N, dtype=float)
		for k in range(0, self.N):
			self.dr[k] = self.r[k+1] - self.r[k]
		self.dmin = self.dr.min()
		self.dmax = self.dr.max()

#-------------------------------------------------------------------#
#		Imposed magnetic field (function of radius only)			#
#-------------------------------------------------------------------#
class Bfield():
	def __init__(self, Btype, B0, grid):
		self.Btype = Btype						# 'uniform', 'dipole' or None
		self.B0    = B0							# magnitude
		self.r0    = grid.r0					# initial radius [real]
		self.rf    = grid.rf					# final radius [real]
		self.r     = grid.r						# radial grid

		if Btype == 'uniform':
			self.uniform()
		elif Btype == 'dipole':
			self.dipole()

	def uniform(self):
		B0 = self.B0
		r  = self.r
		self.Br   = B0 * np.ones_like(r)		# Br = 1
		self.DrBr = np.zeros_like(r)			# dBr/dr = 0
		self.Bs   = -B0 * np.ones_like(r)		# Bs = -1
		self.DrBs = np.zeros_like(r)			# dBs/dr = 0

	def dipole(self):
		if self.r0 == 0:
			print('Error: not support background dipolar magnetic field for full sphere')
			return

		B0 = self.B0
		r = self.r
		self.Br   = B0 * 1/(r**3)				# Br = 1/r^3
		self.DrBr = -B0 * 3/(r**4)				# dBr/dr = -3/r^4
		self.Bs   = B0 * 1/2/(r**3)				# Bs = 1/(2r^3)
		self.DrBs = -B0 * 3/2/(r**4)			# dBs/dr = -3/(2r^4)

#-------------------------------------------------------------------#
#					Finite difference approximation					#
#-------------------------------------------------------------------#
def fd_deriv(order, x, i):
	pts = (order+1)//2*2+1

	i0, i1 = i-pts//2, i+pts//2+1
	if i0 < 0:
		if order%2 == 0:
			pts += 1
		i0, i1 = i, i+pts
	if i1 > x.size:
		if order%2 == 0:
			pts += 1
		i0, i1 = i-pts+1, i+1
	stencil = x[i0:i1] - x[i]

	A = np.zeros((pts, pts), dtype=float)
	b = np.zeros(pts, dtype=float)
	for j in range(pts):
		A[j, :] = stencil**j / math.factorial(j)
	b[order] = 1

	return np.linalg.solve(A, b)

def fd_deriv_row(order, x, i):
	pts = (order+1)//2*2+1

	i0, i1 = i-pts//2, i+pts//2+1
	if i0 < 0:
		if order%2 == 0:
			pts += 1
		i0, i1 = i, i+pts
	if i1 > x.size:
		if order%2 == 0:
			pts += 1
		i0, i1 = i-pts+1, i+1
	stencil = x[i0:i1] - x[i]

	A = np.zeros((pts, pts), dtype=float)
	b = np.zeros(pts, dtype=float)
	for j in range(pts):
		A[j, :] = stencil**j / math.factorial(j)
	b[order] = 1

	coef = np.zeros(x.size, dtype=float)
	coef[i0:i1] = np.linalg.solve(A, b)
	return coef

# Filling the bulk of the Laplacian matrix, without the term in l(l+1)/r^2.
# Output : Laplacian matrix has (N-1)*(N-1) elements.
def mat_lapla(grid):
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)
	# Bulk of the matrix with all the coefficients
	for i in range(0, N-1):
		# i+1: from 1 to N-1
		Dr    = fd_deriv(1, grid.r, i+1)
		D2r   = fd_deriv(2, grid.r, i+1)
		ri    = grid.r[i+1]
		D2r_r = D2r + 2/ri * Dr
		M[i, i:i+3] = D2r_r
	return M[:N-1, 1:N]

# Filling the bulk of the matrix of the first derivative, used for operator L1 and L_{-1}.
# Output : matrix with (N-1)*(N-1) elements.
def mat_deriv1(grid):
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)
	# Bulk of the matrix with all the coefficients
	for i in range(0, N-1):
		# i+1: from 1 to N-1
		Dr = fd_deriv(1, grid.r, i+1)
		M[i, i:i+3] = Dr
	return M[:N-1, 1:N]

# Filling the matrix of the biharmonic operator.
# l      : harmonic degree [integer],
# bc_i   : 0 = BC in r=0 ; 1 = FS to ICB ; 2 = NS to ICB,
# bc_o   : 1 = FS to CMB ; 2 = NS to CMB,
# Output : operator matrix with (N-1)*(N-1) elements.
def mat_bilapla_pol(grid, l, bc_i, bc_o):
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)

	for i in range(N+1):
		ri = grid.r[i]
		Dl2_1 = fd_deriv_row(4, grid.r, i)
		Dl2_2 = 4/ri * fd_deriv_row(3, grid.r, i)
		Dl2_3 = -2*l*(l+1)/ri/ri * fd_deriv_row(2, grid.r, i)
		Dl2_4 = (l-1)*l*(l+1)*(l+2)/ri**4

		M[i] = Dl2_1 + Dl2_2 + Dl2_3
		M[i, i] += Dl2_4

	return M[1:-1, 1:-1]

# Filling the Laplacian matrix for the toroidal.
# lapla  : bulk of the Laplacian matrix, without l(l+1)/r^2, (N-1)*(N-1) elements,
# l      : harmonic degree [integer],
# bc_i   : 0 = BC in r=0 ; 1 = FS to ICB ; 2 = NS to ICB,
# bc_o   : 1 = FS to CMB ; 2 = NS to CMB,
# Output : Laplacian matrix, with nb of elements between (N-1)^2 and (N+1)^2.
def mat_lapla_tor(lapla, grid, l, bc_i, bc_o):
	N = grid.N
	M = lapla - (l*(l+1)) * np.diag(1/grid.r[1:-1]**2)		# (N-1) * (N-1)
	# BC to ICB
	if bc_i == 1:
		M = np.vstack([np.zeros((1, N-1), dtype=float), M])
		M = np.hstack([np.zeros((N, 1), dtype=float), M])
		# Lapla in r0
		ri      = grid.r[0]
		h       = grid.dr[0]
		M[0, 0] = 2/(ri**2) - 2/(h**2) - 2/(ri*h) - l*(l+1)/(ri**2)
		M[0, 1] = 2/(h**2)
		# Missing coefficient at r1
		i = 0
		Dr    = fd_deriv(1, grid.r, i+1)
		D2r   = fd_deriv(2, grid.r, i+1)
		ri    = grid.r[i+1]
		D2r_r = D2r + 2/ri * Dr
		M[1, 0] = D2r_r[0]
	# BC to CMB
	if bc_o == 1:
		nb_row, nb_col = M.shape
		M = np.vstack([M, np.zeros((1, nb_col), dtype=float)])
		M = np.hstack([M, np.zeros((nb_row+1, 1), dtype=float)])
		# Lapla in rN
		ri        = grid.r[-1]
		h         = grid.dr[-1]
		M[-1, -1] = 2/(ri**2) - 2/(h**2) + 2/(ri*h) - l*(l+1)/(ri**2)
		M[-1, -2] = 2/(h**2)
		# Missing coefficient at rN-1
		i = N-2
		Dr        = fd_deriv(1, grid.r, i+1)
		D2r       = fd_deriv(2, grid.r, i+1)
		ri        = grid.r[i+1]
		D2r_r     = D2r + 2/ri * Dr
		M[-2, -1] = D2r_r[2]
	return M

#-------------------------------------------------------------------#
#						Temperature matrix							#
#-------------------------------------------------------------------#
# Filling of the Laplacian matrix for temperature.
# lapla     : bulk of the Laplacian matrix, without l(l+1)/r^2, (N-1)*(N-1) elements,
# grid      : radial grid class,
# l         : harmonic degree> 0,
# bc_i_temp : 0 = BC at r=0 ; 1 = Flux imposes zero on the ICB ; 2 = T imposed zero on the ICB,
# bc_o_temp : 1 = Flow imposed on the CMB ; 2 = T imposed on the CMB.
# Output    : Laplacian matrix whose number of elements depends on the BCs.
def mat_lapla_temp(lapla, grid, l, bc_i_temp, bc_o_temp):
	N = grid.N
	M = lapla - (l*(l+1)) * np.diag(1/grid.r[1:-1]**2)		# (N-1) * (N-1)
	# BC to ICB
	if bc_i_temp == 1:
		M = np.vstack([np.zeros((1, N-1), dtype=float), M])
		M = np.hstack([np.zeros((N, 1), dtype=float), M])
		# Lapla in r0
		ri      = grid.r[0]
		h       = grid.dr[0]
		M[0, 0] = -2/(h**2) - l*(l+1)/(ri**2)
		M[0, 1] = 2/(h**2)
		# Missing coefficient at r1
		i = 0
		Dr    = fd_deriv(1, grid.r, i+1)
		D2r   = fd_deriv(2, grid.r, i+1)
		ri    = grid.r[i+1]
		D2r_r = D2r + 2/ri * Dr
		M[1, 0] = D2r_r[0]
	# BC to CMB
	if bc_o_temp == 1:
		nb_row, nb_col = M.shape
		M = np.vstack([M, np.zeros((1, nb_col), dtype=float)])
		M = np.hstack([M, np.zeros((nb_row+1, 1), dtype=float)])
		# Lapla in rN
		ri        = grid.r[-1]
		h         = grid.dr[-1]
		M[-1, -1] = -2/(h**2) - l*(l+1)/(ri**2)
		M[-1, -2] = 2/(h**2)
		# Missing coefficient at rN-1
		i = N-2
		Dr        = fd_deriv(1, grid.r, i+1)
		D2r       = fd_deriv(2, grid.r, i+1)
		ri        = grid.r[i+1]
		D2r_r     = D2r + 2/ri * Dr
		M[-2, -1] = D2r_r[2]
	return M

# Filling of the matrix of operators L1 and L_{-1} for the poloidal.
# d1          : matrix of the first derivative, with (N-1) * (N-1) elements,
# grid        : radial grid class,
# l           : harmonic degree of the equation [integer],
# couple      : index to choose the operator. 1 = L1 and -1 = L_{-1},
# coeff_spec1 : spectral coefficient of the term 1/r,
# coeff_spec2 : spectral coefficient of the term d/dr,
# bc_i        : 0 = BC in r=0 ; 1 = FS to ICB ; 2 = NS to ICB,
# bc_o        : 1 = FS to CMB ; 2 = NS to CMB.
# Output      : with nb of elements between (N-1)^2 and (N+1)*(N-1).
def mat_L_pol(d1, grid, l, couple, coeff_spec1, coeff_spec2, bc_i, bc_o):
	# Spectral coefficient of the Q3 operator
	if   couple == 1:					# l = l-1, couple=1
		a =  l*(l+2) * coeff_spec1			# a = (l-1)*(l+1) * (l-1)*\alpha_{l-1}^l
		b = -l*(l+2) * coeff_spec2			# b = -(l-1)*(l+1) * \alpha_{l-1}^l
	elif couple == -1:					# l = l+1, couple=-1
		a = (l**2-1) * coeff_spec1			# a = l*(l+2) * (-(l+2)*\alpha_{l+1}^l)
		b = (1-l**2) * coeff_spec2			# b = -l*(l+2) * \alpha_{l+1}^l

	# Initialization
	N = grid.N
	M = b*d1 + a*np.diag(1/grid.r[1:-1])

	# BC to ICB
	if bc_i == 1:
		M = np.vstack([np.zeros((1, N-1), dtype=float), M])
		M[0, 0] = b/grid.dr[0]

	# BC to CMB
	if bc_o == 1:
		M = np.vstack([M, np.zeros((1, N-1), dtype=float)])
		M[-1, -1] = b/grid.dr[-1]

	return M

# Filling the matrix of operators L1 and L_{-1} for the toroidal.
# d1          : matrix of the first derivative, with (N-1) * (N-1) elements,
# grid        : radial grid class,
# l           : harmonic degree of the equation [integer],
# couple      : index to choose the operator. 1 = L1 and -1 = L_{-1},
# coeff_spec1 : spectral coefficient of the term 1/r,
# coeff_spec2 : spectral coefficient of the term d/dr,
# bc_i        : 0 = BC in r=0 ; 1 = FS to ICB ; 2 = NS to ICB,
# bc_o        : 1 = FS to CMB ; 2 = NS to CMB.
# Output      : matrix (N-1)^2 and (N-1)*(N+1)
def mat_L_tor(d1, grid, l, couple, coeff_spec1, coeff_spec2, bc_i, bc_o):
	# Spectral coefficients
	if   couple == 1:					# l = l-1, couple=1
		a =  l*(l+2) * coeff_spec1			# a = (l-1)*(l+1) * (l-1)*\alpha_{l-1}^l
		b = -l*(l+2) * coeff_spec2			# b = -(l-1)*(l+1) * \alpha_{l-1}^l
	elif couple == -1:					# l = l+1, couple=-1
		a = (l**2-1) * coeff_spec1			# a = l*(l+2) * (-(l+2)*\alpha_{l+1}^l)
		b = (1-l**2) * coeff_spec2			# b = -l*(l+2) * \alpha_{l+1}^l

	# Initialization
	N = grid.N
	M = b*d1 + a*np.diag(1/grid.r[1:-1])

	# Boundary conditions
	if bc_i == 1:
		# Missing coefficient at r = r1
		M = np.hstack([np.zeros((N-1, 1), dtype=float), M])
		i = 0
		Dr = fd_deriv(1, grid.r, i+1)
		M[0, 0] = b * Dr[0]

	if bc_o == 1:
		# Missing coefficient at r = rN-1
		M = np.hstack([M, np.zeros((N-1, 1), dtype=float)])
		i = N-2
		Dr = fd_deriv(1, grid.r, i+1)
		M[-1, -1] = b * Dr[-1]

	return M

#-------------------------------------------------------------------#
#						Magnetic field matrix						#
#-------------------------------------------------------------------#
# Filling the matrix of operators A_{l-1}*L_{l-1}^G and A_{l+1}*L_{l+1}^G for the poloidal_b.
# grid        : radial grid class,
# Bfield : imposed magnetic field class,
# l           : harmonic degree of the equation [integer],
# couple      : index to choose the operator. 1 = L_{l-1}^G and -1 = L_{l+1}^G,
# coeff_A     : A_{l-1} (couple=1) or A_{l+1} (couple=-1).
# Output      : matrix (N-1)*(N+1)
def mat_pol_polb(grid, Bfield, l, couple, coeff_A):
	# Spectral coefficients
	if   couple == 1:
		a = l*(l-1)
		b = l
	elif couple == -1:
		a = (l+1)*(l+2)
		b = -(l+1)

	# Initialization
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)

	for i in range(N+1):
		ri   = grid.r[i]
		Br   = Bfield.Br[i]
		DrBr = Bfield.DrBr[i]
		Bs   = Bfield.Bs[i]
		DrBs = Bfield.DrBs[i]

		LG_1 = a/ri**3 * (Br - ri*DrBr - b*Bs)
		LG_2 = 1/ri**2 * (-a*Br + 2*(ri*DrBr + b*Bs)) * fd_deriv_row(1, grid.r, i)
		LG_3 = 1/ri * (3*Br + ri*DrBr + b*Bs) * fd_deriv_row(2, grid.r, i)
		LG_4 = Br * fd_deriv_row(3, grid.r, i)

		M[i] = LG_2 + LG_3 + LG_4
		M[i, i] += LG_1

	return coeff_A*a * M[1:-1]

# Filling the bulk of the L_l^c matrix.
# Output : L_l^c matrix has (N-1)*(N-1) elements.
# grid   : radial grid class,
# Bfield : imposed magnetic field class,
# l      : harmonic degree [integer].
def mat_pol_torb(grid, Bfield, l):
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)

	for i in range(N+1):
		ri   = grid.r[i]
		Br   = Bfield.Br[i]
		DrBr = Bfield.DrBr[i]
		Bs   = Bfield.Bs[i]
		DrBs = Bfield.DrBs[i]

		Llc_1 = Br * fd_deriv_row(2, grid.r, i)
		Llc_2 = (DrBr + 2*Br/ri) * fd_deriv_row(1, grid.r, i)
		Llc_3 = DrBr/ri - l*(l+1) * (-Bs/ri**2 + DrBs/ri)

		M[i] = Llc_1 + Llc_2
		M[i, i] += Llc_3

		return M[1:-1, 1:-1]

# Filling the bulk of the L_l^G matrix.
# Output : L_l^G matrix has (N-1)*(N+1), N*(N+1) or (N+1)*(N+1) elements.
# grid   : radial grid class,
# Bfield : imposed magnetic field class,
# l      : harmonic degree [integer],
# bc_i   : 0 = BC in r=0 ; 1 = FS to ICB ; 2 = NS to ICB,
# bc_o   : 1 = FS to CMB ; 2 = NS to CMB.
def mat_tor_polb(grid, Bfield, l, bc_i, bc_o):
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)

	for i in range(N+1):
		ri   = grid.r[i]
		Br   = Bfield.Br[i]

		LlG_1 = l*(l+1)/ri**2
		LlG_2 = -2/ri * fd_deriv_row(1, grid.r, i)
		LlG_3 = -fd_deriv_row(2, grid.r, i)

		M[i] = LlG_2 + LlG_3
		M[i, i] += LlG_1
		M[i] *= Br/(l*(l+1))

	i0, i1 = 1, N
	# BC to ICB
	if bc_i == 1:
		i0 = 0
	# BC to CMB
	if bc_o == 1:
		i1 = N+1

	return M[i0:i1]

# Filling the matrix of operators A_{l-1}*L_{l-1}^c and A_{l+1}*L_{l+1}^c for the toroidal_b.
# grid        : radial grid class,
# Bfield      : imposed magnetic field class,
# l           : harmonic degree of the equation [integer],
# couple      : index to choose the operator. 1 = L_{l-1}^c and -1 = L_{l+1}^c,
# coeff_A     : A_{l-1} (couple=1) or A_{l+1} (couple=-1),
# bc_i        : 0 = BC in r=0 ; 1 = FS to ICB ; 2 = NS to ICB,
# bc_o        : 1 = FS to CMB ; 2 = NS to CMB.
# Output      : matrix (N-1)*(N-1), N*(N-1) or (N+1)*(N-1)
def mat_tor_torb(grid, Bfield, l, couple, coeff_A, bc_i, bc_o):
	# Spectral coefficients
	if   couple == 1:
		a = l*(l-1)
		b = l
	elif couple == -1:
		a = (l+1)*(l+2)
		b = -(l+1)

	# Initialization
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)

	for i in range(N+1):
		ri   = grid.r[i]
		Br   = Bfield.Br[i]
		Bs   = Bfield.Bs[i]

		Lc_1 = (Br + b*Bs) / ri
		Lc_2 = Br * fd_deriv_row(1, grid.r, i)

		M[i] = Lc_2
		M[i, i] += Lc_1

	i0, i1 = 1, N
	# BC to ICB
	if bc_i == 1:
		i0 = 0
	# BC to CMB
	if bc_o == 1:
		i1 = N+1

	return coeff_A*a * M[i0:i1, 1:-1]

# Filling the matrix of operators A_{l-1}*L_{l-1}^P and A_{l+1}*L_{l+1}^P for the poloidal.
# grid        : radial grid class,
# Bfield      : imposed magnetic field class,
# l           : harmonic degree of the equation [integer],
# couple      : index to choose the operator. 1 = L_{l-1}^P and -1 = L_{l+1}^P,
# coeff_A     : A_{l-1} (couple=1) or A_{l+1} (couple=-1).
# Output      : matrix (N+1)*(N-1)
def mat_polb_pol(grid, Bfield, l, couple, coeff_A):
	# Spectral coefficients
	if   couple == 1:
		a = l*(l-1)
		b = l
	elif couple == -1:
		a = (l+1)*(l+2)
		b = -(l+1)

	# Initialization
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)

	for i in range(N+1):
		ri   = grid.r[i]
		Br   = Bfield.Br[i]
		Bs   = Bfield.Bs[i]

		LP_1 = (Br + b*Bs) / ri
		LP_2 = Br * fd_deriv_row(1, grid.r, i)

		M[i] = LP_2
		M[i, i] += LP_1

	return coeff_A*a * M[:, 1:-1]

# Filling the Laplacian matrix for the poloidal_b.
# grid   : radial grid class,
# l      : harmonic degree [integer].
def mat_polb_polb(grid, l):
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)

	for i in range(N+1):
		ri  = grid.r[i]
		Dr  = fd_deriv_row(1, grid.r, i)
		D2r = fd_deriv_row(2, grid.r, i)

		M[i] = D2r + 2/ri * Dr
		M[i, i] -= l*(l+1)/ri**2

	return M

# Filling the bulk of the L_l^P matrix.
# Output : L_l^P matrix has (N-1)*(N-1) elements.
# grid   : radial grid class,
# Bfield : imposed magnetic field class,
# l      : harmonic degree [integer].
def mat_torb_pol(grid, Bfield, l):
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)

	for i in range(N+1):
		ri   = grid.r[i]
		Br   = Bfield.Br[i]
		DrBr = Bfield.DrBr[i]
		Bs   = Bfield.Bs[i]
		DrBs = Bfield.DrBs[i]

		LlP_1 = -Br * fd_deriv_row(2, grid.r, i)
		LlP_2 = -(2*Br/ri + DrBr) * fd_deriv_row(1, grid.r, i)
		LlP_3 = -DrBr/ri + l*(l+1) * (-Bs/ri**2 + DrBs/ri)

		M[i] = LlP_1 + LlP_2
		M[i, i] += LlP_3

		return M[1:-1, 1:-1]

# Filling the matrix of operators A_{l-1}*L_{l-1}^w and A_{l+1}*L_{l+1}^w for the poloidal.
# grid        : radial grid class,
# Bfield      : imposed magnetic field class,
# l           : harmonic degree of the equation [integer],
# couple      : index to choose the operator. 1 = L_{l-1}^w and -1 = L_{l+1}^w,
# coeff_A     : A_{l-1} (couple=1) or A_{l+1} (couple=-1),
# bc_i   : 0 = BC in r=0 ; 1 = FS to ICB ; 2 = NS to ICB,
# bc_o   : 1 = FS to CMB ; 2 = NS to CMB.
# Output      : matrix (N-1)*(N-1), (N-1)*N or (N-1)*(N+1)
def mat_torb_tor(grid, Bfield, l, couple, coeff_A, bc_i, bc_o):
	# Spectral coefficients
	if   couple == 1:
		a = l*(l-1)
		b = l
	elif couple == -1:
		a = (l+1)*(l+2)
		b = -(l+1)

	# Initialization
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)

	for i in range(N+1):
		ri   = grid.r[i]
		Br   = Bfield.Br[i]
		DrBr = Bfield.DrBr[i]
		Bs   = Bfield.Bs[i]

		Lw_1 = (Br + b*Bs) / ri + DrBr
		Lw_2 = Br * fd_deriv_row(1, grid.r, i)

		M[i] = Lw_2
		M[i, i] += Lw_1

	i0, i1 = 1, N
	# BC to ICB
	if bc_i == 1:
		i0 = 0
	# BC to CMB
	if bc_o == 1:
		i1 = N+1

	return coeff_A*a * M[1:-1, i0:i1]

# Filling the Laplacian matrix for the toroidal_b.
# grid   : radial grid class,
# l      : harmonic degree [integer].
def mat_torb_torb(grid, l):
	N = grid.N
	M = np.zeros((N+1, N+1), dtype=float)

	for i in range(N+1):
		ri  = grid.r[i]
		Dr  = fd_deriv_row(1, grid.r, i)
		D2r = fd_deriv_row(2, grid.r, i)

		M[i] = D2r + 2/ri * Dr
		M[i, i] -= l*(l+1)/ri**2

	return M[1:-1, 1:-1]



# def fd_deriv(order, x, i):
# 	# second order FD formulas:
# 	dl = x[i]-x[i-1]
# 	du = x[i+1]-x[i]
# 	if order == 1:
# 		Dx = np.zeros(3, dtype=float)
# 		Dx[0] = -du/(dl*(dl + du))
# 		Dx[1] = (-dl + du)/(dl*du)
# 		Dx[2] = dl/(du*(dl + du))
# 		return Dx
# 	elif order == 2:
# 		D2x = np.zeros(3, dtype=float)
# 		D2x[0] = 2/(dl*(dl + du))
# 		D2x[1] = -2/(dl*du)
# 		D2x[2] = 2/(du*(dl + du))
# 		return D2x

# def mat_lapla(grid):
# 	N = grid.N
# 	M = np.zeros((N-1, N-1), dtype=float)
# 	# Bulk of the matrix with all the coefficients
# 	for i in range(1, N-2):
# 		# i+1: from 2 to N-2
# 		Dr    = fd_deriv(1, grid.r, i+1)
# 		D2r   = fd_deriv(2, grid.r, i+1)
# 		ri    = grid.r[i+1]
# 		D2r_r = D2r + 2/ri * Dr
# 		M[i, i-1:i+2] = D2r_r
# 	# Order 2 on the edges for the poloidal
# 		# In r1
# 	i = 0
# 	Dr    = fd_deriv(1, grid.r, i+1)
# 	D2r   = fd_deriv(2, grid.r, i+1)
# 	ri    = grid.r[i+1]
# 	D2r_r = D2r + 2/ri * Dr
# 	M[i, i:i+2] = D2r_r[1:]
# 		# In rN-1
# 	i = N-2
# 	Dr    = fd_deriv(1, grid.r, i+1)
# 	D2r   = fd_deriv(2, grid.r, i+1)
# 	ri    = grid.r[i+1]
# 	D2r_r = D2r + 2/ri * Dr
# 	M[i, i-1:i+1] = D2r_r[:-1]
# 	return M

# def mat_deriv1(grid):
# 	N = grid.N
# 	M = np.zeros((N-1, N-1), dtype=float)
# 	# Bulk of the matrix with all the coefficients
# 	for i in range(1, N-2):
# 		Dr = fd_deriv(1, grid.r, i+1)
# 		M[i, i-1:i+2] = Dr
# 	# Order 2 on the edges for the poloidal
# 		# In r1
# 	i = 0
# 	Dr = fd_deriv(1, grid.r, i+1)
# 	M[i, i:i+2] = Dr[1:]
# 		# In rN-1
# 	i = N-2
# 	Dr = fd_deriv(1, grid.r, i+1)
# 	M[i, i-1:i+1] = Dr[:-1]
# 	return M

# def mat_bilapla_pol(lapla, grid, l, bc_i, bc_o):
# 	N  = grid.N
# 	M  = lapla - (l*(l+1)) * np.diag(1/grid.r[1:-1]**2)		# (N-1) * (N-1)
# 	M1 = np.vstack([np.zeros((1, N-1), dtype=float), M, np.zeros((1, N-1), dtype=float)])	# (N+1) * (N-1)
# 	M2 = np.hstack([np.zeros((N-1, 1), dtype=float), M, np.zeros((N-1, 1), dtype=float)])	# (N-1) * (N+1)

# 	# BC at the ICB or at the center
# 	i = 0
# 	Dr    = fd_deriv(1, grid.r, i+1)
# 	D2r   = fd_deriv(2, grid.r, i+1)
# 	ri    = grid.r[i+1]
# 	D2r_r = D2r + 2/ri * Dr
# 	if   bc_i == 0:
# 		M1 = np.delete(M1, 0, axis=0)
# 		M2 = np.delete(M2, 0, axis=1)	# Delete the first column
# 	elif bc_i == 1:
# 		k0       = 2/(grid.r[0]*grid.dr[0])
# 		M1[0, 0] = k0
# 		M2[0, 0] = D2r_r[0]
# 	elif bc_i == 2:
# 		k0       = 2/(grid.dr[0]**2)
# 		M1[0, 0] = k0
# 		M2[0, 0] = D2r_r[0]

# 	# BC at the CMB
# 	i = N-2
# 	Dr    = fd_deriv(1, grid.r, i+1)
# 	D2r   = fd_deriv(2, grid.r, i+1)
# 	ri    = grid.r[i+1]
# 	D2r_r = D2r + 2/ri * Dr
# 	M2[-1, -1] = D2r_r[-1]
# 	if   bc_o == 1:
# 		kN         = -2/(grid.r[-1]*grid.dr[-1])
# 		M1[-1, -1] = kN
# 	elif bc_o == 2:
# 		kN         = 2/(grid.dr[-1]**2)
# 		M1[-1, -1] = kN

# 	M = np.dot(M2, M1)
# 	return M

# def mat_lapla_tor(lapla, grid, l, bc_i, bc_o):
# 	N = grid.N
# 	M = lapla - (l*(l+1)) * np.diag(1/grid.r[1:-1]**2)		# (N-1) * (N-1)
# 	# BC to ICB
# 	if bc_i == 1:
# 		M = np.vstack([np.zeros((1, N-1), dtype=float), M])
# 		M = np.hstack([np.zeros((N, 1), dtype=float), M])
# 		# Lapla in r0
# 		h       = grid.dr[0]
# 		M[0, 0] = 2/(grid.r[0]**2) - 2/h**2 - 2/(h*grid.r[0]) - l*(l+1)/(grid.r[0]**2)
# 		M[0, 1] = 2/h**2
# 		# Missing coefficient at r1
# 		i = 0
# 		Dr    = fd_deriv(1, grid.r, i+1)
# 		D2r   = fd_deriv(2, grid.r, i+1)
# 		D2r_r = D2r + 2/grid.r[i+1] * Dr
# 		M[1, 0] = D2r_r[0]
# 	# BC to CMB
# 	if bc_o == 1:
# 		nb_l,nb_col = M.shape
# 		M = np.vstack([M, np.zeros((1, nb_col), dtype=float)])
# 		M = np.hstack([M, np.zeros((nb_l+1, 1), dtype=float)])
# 		# Lapla in rN
# 		h         = grid.dr[-1]
# 		M[-1, -1] = 2/(grid.r[-1]**2) - 2/h**2 + 2/(h*grid.r[-1]) - l*(l+1)/(grid.r[-1]**2)
# 		M[-1, -2] = 2/h**2
# 		# Missing coefficient at rN-1
# 		i = N-2
# 		Dr        = fd_deriv(1, grid.r, i+1)
# 		D2r       = fd_deriv(2, grid.r, i+1)
# 		D2r_r     = D2r + 2/grid.r[i+1] * Dr
# 		M[-2, -1] = D2r_r[2]
# 	return M

# # Filling the bulk of the Laplacian matrix, without the term in l(l+1)/r^2.
# # Output : Laplacian matrix has (N-1)*(N-1) elements.
# def mat_lapla(grid):
# 	d_dr   = FinDiff(0, grid.r, 1, acc=2)
# 	d2_dr2 = FinDiff(0, grid.r, 2, acc=2)

# 	Dr   = d_dr.matrix(grid.r.shape).toarray()
# 	D2r  = d2_dr2.matrix(grid.r.shape).toarray()
# 	Diag = np.diag(2/grid.r)
# 	D2r_r = D2r + Diag @ Dr

# 	return D2r_r[1:-1, 1:-1]

# # Filling the bulk of the matrix of the first derivative, used for operator L1 and L_{-1}.
# # Output : matrix with (N-1)*(N-1) elements.
# def mat_deriv1(grid):
# 	d_dr = FinDiff(0, grid.r, 1, acc=2)
# 	Dr = d_dr.matrix(grid.r.shape).toarray()
# 	return Dr[1:-1, 1:-1]

# # Filling the matrix of the biharmonic operator.
# # lapla  : bulk of the Laplacian matrix, without l(l+1)/r^2, (N-1)*(N-1) elements,
# # l      : harmonic degree [integer],
# # bc_i   : 0 = BC in r=0 ; 1 = FS to ICB ; 2 = NS to ICB,
# # bc_o   : 1 = FS to CMB ; 2 = NS to CMB,
# # Output : operator matrix with (N-1)*(N-1) elements.
# def mat_bilapla_pol(lapla, grid, l, bc_i, bc_o):
# 	N  = grid.N
# 	M  = lapla - (l*(l+1)) * np.diag(1/grid.r[1:-1]**2)		# (N-1) * (N-1)
# 	M1 = np.vstack([np.zeros((1, N-1), dtype=float), M, np.zeros((1, N-1), dtype=float)])	# (N+1) * (N-1)
# 	M2 = np.hstack([np.zeros((N-1, 1), dtype=float), M, np.zeros((N-1, 1), dtype=float)])	# (N-1) * (N+1)

# 	# BC at the ICB or at the center
# 	i = 0
# 	Dr    = fd_deriv(1, grid.r, i+1)
# 	D2r   = fd_deriv(2, grid.r, i+1)
# 	ri    = grid.r[i+1]
# 	D2r_r = D2r + 2/ri * Dr
# 	M2[0, 0] = D2r_r[0]
# 	if   bc_i == 0:
# 		k0 = 0
# 	elif bc_i == 1:
# 		k0 = 2/(grid.r[0]*grid.dr[0])
# 	elif bc_i == 2:
# 		k0 = 2/(grid.dr[0]**2)
# 	M1[0, 0] = k0

# 	# BC at the CMB
# 	i = N-2
# 	Dr    = fd_deriv(1, grid.r, i+1)
# 	D2r   = fd_deriv(2, grid.r, i+1)
# 	ri    = grid.r[i+1]
# 	D2r_r = D2r + 2/ri * Dr
# 	M2[-1, -1] = D2r_r[-1]
# 	if   bc_o == 1:
# 		kN = -2/(grid.r[-1]*grid.dr[-1])
# 	elif bc_o == 2:
# 		kN = 2/(grid.dr[-1]**2)
# 	M1[-1, -1] = kN

# 	M = np.dot(M2, M1)
# 	return M

# from findiff import FinDiff

# # Filling the matrix of the biharmonic operator.
# # l      : harmonic degree [integer],
# # bc_i   : 0 = BC in r=0 ; 1 = FS to ICB ; 2 = NS to ICB,
# # bc_o   : 1 = FS to CMB ; 2 = NS to CMB,
# # Output : operator matrix with (N-1)*(N-1) elements.
# def mat_bilapla_pol(grid, l, bc_i, bc_o):
# 	d_dx   = FinDiff(0, grid.r, 1, acc=2)
# 	d2_dx2 = FinDiff(0, grid.r, 2, acc=2)
# 	d3_dx3 = FinDiff(0, grid.r, 3, acc=2)
# 	d4_dx4 = FinDiff(0, grid.r, 4, acc=2)

# 	r = grid.r;	shape = r.shape
# 	Dl2_1 = d4_dx4.matrix(shape).toarray()
# 	Dl2_2 = 4*np.diag(1/r) @ d3_dx3.matrix(shape).toarray()
# 	Dl2_3 = -l*(l+1)*np.diag(1/r/r) @ d2_dx2.matrix(shape).toarray()
# 	Dl2_4 = -2*l*(l+1)*np.diag(1/r**3) @ d_dx.matrix(shape).toarray()
# 	Dl2_5 = (l-1)*l*(l+1)*(l+2)*np.diag(1/r**4)

# 	M = Dl2_1 + Dl2_2 + Dl2_3 + Dl2_4 + Dl2_5
# 	return M[1:-1, 1:-1]