# General packages
import sys

# Scientific packages
import numpy as np

# Hand-made packages
import params
import sub_op_irreg as sub_op
import sub_eig_irreg as sub_eig
import pyxshells as px

print("""\n################################################
\tConversion from MAGSINGE to XSHELLS
################################################
""")

#-----------------------------------------------------------------------
# Sub function
#-----------------------------------------------------------------------
def vel2xshells(grid, pol, tor, bc_i, bc_o, filename):
	# Init
	irs, ire = 0, len(grid)-1			# define radial extent
	grid_xshells = px.Grid(grid)		# build the radial grid
	U = px.PolTor(grid_xshells, sh)		# build a PolTor field for xshells from the grid and the sht
	U.alloc(irs, ire)					# alloc memory for the field
	# Assign
	for ir in range(irs, ire+1):
		U.set_pol(ir, pol[ir, :])
		U.set_tor(ir, tor[ir, :])
	# BC
	if (bc_i == 0) or (bc_i == 1):		# Stress-free
		BCI = 2
	elif bc_i == 2:						# No-slip
		BCI = 1
	if bc_o == 1:
		BCO = 2
	elif bc_o == 2:
		BCO = 1
	U.set_BC(BCI, BCO)
	# Save
	U.tofile(filename)		# write to file

def temp2xshells(grid, temp, bc_i, bc_o, filename):
	# Init
	irs, ire = 0, len(grid)-1			# define radial extent
	grid_xshells = px.Grid(grid)		# build the radial grid
	T = px.ScalarSH(grid_xshells, sh)	# build a scalar field for xshells from the grid and the sht
	T.alloc(irs, ire)					# alloc memory for the field
	# Assign
	for ir in range(irs, ire+1):
		T.set_sh(temp[ir, :], ir)
	# BC
	if (bc_i == 0) or (bc_i == 1):		# Fixed flux
		BCI = 2
	elif bc_i == 2:						# Fixed temperature
		BCI = 1
	if bc_o == 1:
		BCO = 2
	elif bc_o ==2:
		BCO = 1
	T.set_BC(BCI,BCO)
	# Save
	T.tofile(filename)		# write to file

def mag2xshells(grid, polb, torb, filename):
	# Init
	irs, ire = 0, len(grid)-1			# define radial extent
	grid_xshells = px.Grid(grid)		# build the radial grid
	B = px.PolTor(grid_xshells, sh)		# build a PolTor field for xshells from the grid and the sht
	B.alloc(irs, ire)					# alloc memory for the field
	# Assign
	for ir in range(irs, ire+1):
		B.set_pol(ir, polb[ir, :])
		B.set_tor(ir, torb[ir, :])
	# BC
	BCI, BCO = 3, 3
	B.set_BC(BCI, BCO)
	# Save
	B.tofile(filename)		# write to file

#-----------------------------------------------------------------------
# Main script
#-----------------------------------------------------------------------
if __name__ == "__main__":
	#-----------------------------------------------------------------------
	# Init
	#-----------------------------------------------------------------------
	print('o Radial grid')
	if params.reg == 'reg':
		grid = sub_op.radial_grid(params.r0, params.rf, params.N)
		grid.mesh_reg()
	elif params.reg == 'irreg':
		grid = sub_op.radial_grid(params.r0, params.rf, params.N)
		grid.mesh_irreg(params.nin, params.nout)

	# SHTNS init
	Lmax, sym, sh = sub_eig.set_shtns(params.sym, params.Lmax, params.m)

	# Init temp
	if params.Ra != 0:
		flag_temp = 1
		bc_i_temp, bc_o_temp = params.bc_i_temp, params.bc_o_temp
		bc_i_chi, bc_o_chi = 1, 1		# Default value. Does not play any role because composition is disabled!
		if params.Rac != 0:
			flag_temp = 2
			bc_i_chi, bc_o_chi = params.bc_i_chi, params.bc_o_chi
	else:
		flag_temp = 0
		bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi = 1, 1, 1, 1	# Default value. Does not play any role because temperature and composition are disabled!

	flag_mag = 0		# Default value.
	if params.Btype == 'uniform' or params.Btype == 'dipole':
		flag_mag = 1

	# Spectral params
	pol, tor, temp, chi, polb, torb = sub_eig.nb_eig_vec(params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)

	# Load data in python format
	class solution:
		"Solution class. (To modify in the next version because it is useless)"
	sol = solution()

	title_txt21 = sys.argv[1]		# 'Real_Eigenvec.npy'
	title_txt22 = sys.argv[2]		# 'Imag_Eigenvec.npy'
	vec_real = np.load(title_txt21)
	vec_imag = np.load(title_txt22)
	sol.vec = vec_real + 1j*vec_imag

	#--------
	# Loop on each eigen pair
	#--------
	print("\no Conversion of eigenmodes")
	ind_vec = 0		# index of eigenvector (only 0 available here)
	print('\t* Eigenmode %i' %(ind_vec))
	# Extraction of the eigenvectors for an eigenvalue
	pol.vec = sub_eig.choice_eig(ind_vec, 'pol', sol, flag_temp, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	tor.vec = sub_eig.choice_eig(ind_vec, 'tor', sol, flag_temp, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	if params.Ra != 0:
		temp.vec = sub_eig.choice_eig(ind_vec, 'temp', sol, flag_temp, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
		if params.Rac != 0:
			chi.vec = sub_eig.choice_eig(ind_vec, 'chi', sol, flag_temp, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	if flag_mag == 1:
		polb.vec = sub_eig.choice_eig(ind_vec, 'polb', sol, flag_temp, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
		torb.vec = sub_eig.choice_eig(ind_vec, 'torb', sol, flag_temp, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)

	# Formatting for SHTNS
	pol.vec_shtns = sub_eig.scal_shtns('pol', pol.vec, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	tor.vec_shtns = sub_eig.scal_shtns('tor', tor.vec, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	if params.Ra != 0:
		temp.vec_shtns = sub_eig.scal_shtns('temp', temp.vec, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
		if params.Rac != 0:
			chi.vec_shtns = sub_eig.scal_shtns('chi', chi.vec, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	if flag_mag == 1:
		polb.vec_shtns = sub_eig.scal_shtns('polb', polb.vec, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
		torb.vec_shtns = sub_eig.scal_shtns('torb', torb.vec, params.m, params.N, Lmax, sym, params.bc_i, params.bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	# BC
	# Poloidal
	nb_l, nb_c = pol.vec_shtns.shape
	pol.vec_shtns = np.vstack([np.zeros((1, nb_c)), pol.vec_shtns, np.zeros((1, nb_c))])
	# Toroidal
	nb_l, nb_c = tor.vec_shtns.shape
	if params.bc_i == 0 or params.bc_i == 2:
		tor.vec_shtns = np.vstack([np.zeros((1, nb_c)), tor.vec_shtns])
	if params.bc_o == 2:
		tor.vec_shtns = np.vstack([tor.vec_shtns, np.zeros((1, nb_c))])
	# Temperature
	if params.Ra != 0:
		nb_l, nb_c = temp.vec_shtns.shape
		if bc_i_temp == 0 or bc_i_temp == 2:
			temp.vec_shtns = np.vstack([np.zeros((1, nb_c)), temp.vec_shtns])
		if bc_o_temp == 2:
			temp.vec_shtns = np.vstack([temp.vec_shtns, np.zeros((1, nb_c))])
		if params.Rac != 0:
			nb_l, nb_c = chi.vec_shtns.shape
			if bc_i_chi == 0 or bc_i_chi == 2:
				chi.vec_shtns = np.vstack([np.zeros((1, nb_c)), chi.vec_shtns])
			if bc_o_chi == 2:
				chi.vec_shtns = np.vstack([chi.vec_shtns, np.zeros((1, nb_c))])
	# Toroidal_b
	if flag_mag == 1:
		nb_l, nb_c = torb.vec_shtns.shape
		torb.vec_shtns = np.vstack([np.zeros((1, nb_c)), torb.vec_shtns, np.zeros((1, nb_c))])

	# Towards xshells
	filenameU = 'fieldU' + str(ind_vec) + '.out'
	vel2xshells(grid.r, pol.vec_shtns, tor.vec_shtns, params.bc_i, params.bc_o, filenameU)
	if params.Ra != 0:
		filenameT = 'fieldT' + str(ind_vec) + '.out'
		temp2xshells(grid.r, temp.vec_shtns, bc_i_temp, bc_o_temp, filenameT)
		if params.Rac != 0:
			filenameCHI = 'fieldC' + str(ind_vec) + '.out'
			temp2xshells(grid.r, chi.vec_shtns, bc_i_chi, bc_o_chi, filenameCHI)
	if flag_mag == 1:
		filenameB = 'fieldB' + str(ind_vec) + '.out'
		mag2xshells(grid.r, polb.vec_shtns, torb.vec_shtns, filenameB)