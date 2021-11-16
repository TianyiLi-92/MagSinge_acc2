import numpy as np
import shtns
import sub_op_irreg as sub

#-------------------------------------------------------------------#
#						Init SHTns params							#
#-------------------------------------------------------------------#
def set_shtns(sym, lmax, m):
	#----------------
	# Truncation of spherical harmonics
	#	m even => Lmax odd
	#	m odd  => Lmax even
	#	m = 0  => Lmax even
	#----------------
	if (m%2 == 0) and (m > 0):
		if lmax%2 == 0:
			Lmax = lmax + 1
		else:
			Lmax = lmax
	elif (m%2 == 1) or (m == 0):
		if lmax%2 == 0:
			Lmax = lmax
		else:
			Lmax = lmax + 1

	#----------------
	# SHTNS init
	#----------------
	if m == 0:
		l0 = 1
		mmax_shtns = 0
		sh = shtns.sht(Lmax, mmax_shtns, 1)
	elif m > 0:
		l0 = m
		mmax_shtns = 1
		sh = shtns.sht(Lmax, mmax_shtns, m)
	sh.set_grid()

	return Lmax, sym, sh

#-------------------------------------------------------------------#
#		Useful functions to determine the size of the matrices		#
#-------------------------------------------------------------------#
def nb_eig_vec(m, N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi):
	#----------------------------------------------------------------
	""" Returns various information about the eigenvector for the three scalars.
	Inputs :
		* m         : azimuthal order of the mode,
		* N         : nb of shelf intervals,
		* Lmax      : harmonic degree of truncation [even integer],
		* sym       : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric.
		* bc_i      : 0 = BC in r=0 ; 1 = FS to ICB ; 2 = NS to ICB
		* bc_o      : 1 = FS to CMB ; 2 = NS to CMB,
		* bc_i_temp : 0 = BC in r=0 ; 1 = Flux imposes zero on the ICB ; 2 = T zero at ICB,
		* bc_o_temp : 1 = Flux imposed on the CMB ; 2 = T imposed on the CMB.
	Outputs :
		* pol, tor, temp, polb, torb : classes of the three scalars.
	"""
	#----------------------------------------------------------------
	class scalar:
		"Definition of the scalar class for pol, tor, temp, chi, polb, torb"
	pol  = scalar()
	tor  = scalar()
	temp = scalar()
	chi  = scalar()
	polb = scalar()
	torb = scalar()

	# Number of shelf elements according to BC (except for B)
	pol.nbel  = N-1
	tor.nbel  = N-1
	temp.nbel = N-1
	chi.nbel  = N-1

	if bc_i == 1:
		tor.nbel  += 1
	if bc_o == 1:
		tor.nbel  += 1
	if bc_i_temp == 1:
		temp.nbel += 1
	if bc_o_temp == 1:
		temp.nbel += 1
	if bc_i_chi == 1:
		chi.nbel  += 1
	if bc_o_chi == 1:
		chi.nbel  += 1

	# Number of elements fixed for b (insulating at the CMB and zero at the center)
	polb.nbel = N
	torb.nbel = N-1
	if bc_i != 0:
		polb.nbel += 1

	# even modes
	if sym == 'pos':
	# P_m, P_{m+2}, ... and T_{m+1}, T_{m+3}, ...
	# Pb_{m+1}, Pb_{m+3}, ... and Tb_m, Tb_{m+2}, ...
		if   Lmax%2 == 0 and m == 0:
		# Harmonics start at (P_2, T_1, Pb_1, Tb_2) and end at (P_Lmax, T_{Lmax-1}, Pb_{Lmax-1}, Tb_Lmax)
			pol.Lmin, tor.Lmin, temp.Lmin, chi.Lmin, polb.Lmin, torb.Lmin = 2, 1, 2, 2, 1, 2
			pol.nbl = Lmax//2
			tor.nbl, temp.nbl, chi.nbl, polb.nbl, torb.nbl = pol.nbl, pol.nbl, pol.nbl, pol.nbl, pol.nbl
			pol.L  = np.arange(pol.Lmin, Lmax+1, 2)
			tor.L  = np.arange(tor.Lmin, Lmax, 2)
			temp.L = np.arange(temp.Lmin, Lmax+1, 2)
			chi.L  = np.arange(chi.Lmin, Lmax+1, 2)
			polb.L = np.arange(polb.Lmin, Lmax, 2)
			torb.L = np.arange(torb.Lmin, Lmax+1, 2)
		elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
		# Harmonics start at (P_m, T_{m+1}, Pb_{m+1}, Tb_m) and end at (P_{Lmax-1}, T_Lmax, Pb_Lmax, Tb_{Lmax-1})
			pol.Lmin, tor.Lmin, temp.Lmin, chi.Lmin, polb.Lmin, torb.Lmin = m, m+1, m, m, m+1, m
			pol.nbl  = (Lmax - pol.Lmin + 1)//2
			tor.nbl, temp.nbl, chi.nbl, polb.nbl, torb.nbl = pol.nbl, pol.nbl, pol.nbl, pol.nbl, pol.nbl
			pol.L  = np.arange(pol.Lmin, Lmax, 2)
			tor.L  = np.arange(tor.Lmin, Lmax+1, 2)
			temp.L = np.arange(temp.Lmin, Lmax, 2)
			chi.L  = np.arange(chi.Lmin, Lmax, 2)
			polb.L = np.arange(polb.Lmin, Lmax+1, 2)
			torb.L = np.arange(torb.Lmin, Lmax, 2)

	# odd modes
	elif sym == 'neg':
	# P_{m+1}, P_{m+3}, ... and T_m, T_{m+2}, ...
	# Pb_m, Pb_{m+2}, ... and Tb_{m+1}, Tb_{m+3}, ...
		if   Lmax%2 == 0 and m == 0:
		# Harmonics start at (P_1, T_2, Pb_2, Tb_1) and end at (P_{Lmax-1}, T_Lmax, Pb_Lmax, Tb_{Lmax-1})
			pol.Lmin, tor.Lmin, temp.Lmin, chi.Lmin, polb.Lmin, torb.Lmin = 1, 2, 1, 1, 2, 1
			pol.nbl  = Lmax//2
			tor.nbl, temp.nbl, chi.nbl, polb.nbl, torb.nbl = pol.nbl, pol.nbl, pol.nbl, pol.nbl, pol.nbl
			pol.L  = np.arange(pol.Lmin, Lmax, 2)
			tor.L  = np.arange(tor.Lmin, Lmax+1, 2)
			temp.L = np.arange(temp.Lmin, Lmax, 2)
			chi.L  = np.arange(chi.Lmin, Lmax, 2)
			polb.L = np.arange(polb.Lmin, Lmax+1, 2)
			torb.L = np.arange(torb.Lmin, Lmax, 2)
		elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
		# Harmonics start at P_{m+1} T_m and end at P_Lmax and T_{Lmax-1}
			pol.Lmin, tor.Lmin, temp.Lmin, chi.Lmin, polb.Lmin, torb.Lmin = m+1, m, m+1, m+1, m, m+1
			pol.nbl  = (Lmax - pol.Lmin + 2)//2
			tor.nbl, temp.nbl, chi.nbl, polb.nbl, torb.nbl = pol.nbl, pol.nbl, pol.nbl, pol.nbl, pol.nbl
			pol.L  = np.arange(pol.Lmin, Lmax+1, 2)
			tor.L  = np.arange(tor.Lmin, Lmax, 2)
			temp.L = np.arange(temp.Lmin, Lmax+1, 2)
			chi.L  = np.arange(chi.Lmin, Lmax+1, 2)
			polb.L = np.arange(polb.Lmin, Lmax, 2)
			torb.L = np.arange(torb.Lmin, Lmax+1, 2)

	pol.nbel_vec  = pol.nbel * pol.nbl
	tor.nbel_vec  = tor.nbel * tor.nbl
	temp.nbel_vec = temp.nbel * temp.nbl
	chi.nbel_vec  = chi.nbel * chi.nbl
	polb.nbel_vec = polb.nbel * polb.nbl
	torb.nbel_vec = torb.nbel * torb.nbl

	return pol, tor, temp, chi, polb, torb

def ind_eig_mat(scal_eqn, scal_vec, L_eqn, L_vec, ind_r_eqn, ind_r_vec, flag_temp, pol, tor, temp, chi, polb, torb):
	#----------------------------------------------------------------
	""" Returns positions (I, J) in the matrix A or B, for the scalar (pol, tor, temp, chi, polb, torb).
	Inputs :
		* scal_eqn  : 'pol', 'tor', 'polb', 'torb' for the equation,
		* scal_vec  : 'pol', 'tor', 'polb', 'torb' for the eigenvector,
		* L_eqn     : harmonic degree for the equation,
		* L_vec     : harmonic degree for the eigenvector,
		* ind_r_eqn : index of the radius in the equation,
		* ind_r_vec : index of the radius in the eigenvector,
		* flag_temp : enable or not temperature field,
		* pol, tor, temp, polb, torb : outputs of nb_eig_vec.
	Outputs :
		* I         : row in the matrix,
		* J         : column in the matrix.
	"""
	#----------------------------------------------------------------
	ind_eig_eqn = ind_eig_vec
	I = ind_eig_eqn(scal_eqn, L_eqn, ind_r_eqn, flag_temp, pol, tor, temp, chi, polb, torb)
	J = ind_eig_vec(scal_vec, L_vec, ind_r_vec, flag_temp, pol, tor, temp, chi, polb, torb)
	return I, J

def ind_eig_vec(scal_vec, L_vec, ind_r_vec, flag_temp, pol, tor, temp, chi, polb, torb):
	#----------------------------------------------------------------
	""" Returns the position J for the scalar (pol, tor, temp, chi, polb, torb) in the eigenvector.
	Inputs :
		* scal_vec  : 'pol', 'tor', 'polb', 'torb' for the eigenvector,
		* L_vec     : harmonic degree for the eigenvector,
		* ind_r_vec : index of the radius in the eigenvector,
		* flag_temp : enable or not the temperature field,
		* pol, tor, temp, polb, torb : outputs of nb_eig_vec.
	Outputs :
		* J         : column in the matrix.
	"""
	#----------------------------------------------------------------
	# Disable or not of the thermal part
	if   flag_temp == 0:
		TEMP = 0
	elif flag_temp == 1:
		TEMP = temp.nbel_vec
	elif flag_temp == 2:
		TEMP = temp.nbel_vec + chi.nbel_vec

	# Choice of the eigenvector
	if   scal_vec == 'pol':
		ind_L_vec = np.where(pol.L == L_vec)[0][0]	# Index extraction
		J = ind_r_vec * pol.nbl + ind_L_vec
	elif scal_vec == 'tor':
		ind_L_vec = np.where(tor.L == L_vec)[0][0]
		J = ind_r_vec * tor.nbl + ind_L_vec + pol.nbel_vec
	elif scal_vec == 'temp':
		ind_L_vec = np.where(temp.L == L_vec)[0][0]
		J = ind_r_vec * temp.nbl + ind_L_vec + pol.nbel_vec + tor.nbel_vec
	elif scal_vec == 'chi':
		ind_L_vec = np.where(chi.L == L_vec)[0][0]
		J = ind_r_vec * chi.nbl + ind_L_vec + pol.nbel_vec + tor.nbel_vec + temp.nbel_vec
	elif scal_vec == 'polb':
		ind_L_vec = np.where(polb.L == L_vec)[0][0]
		J = ind_r_vec * polb.nbl + ind_L_vec + pol.nbel_vec + tor.nbel_vec + TEMP
	elif scal_vec == 'torb':
		ind_L_vec = np.where(torb.L == L_vec)[0][0]
		J = ind_r_vec * torb.nbl + ind_L_vec + pol.nbel_vec + tor.nbel_vec + TEMP + polb.nbel_vec

	return J

def coeff_stdY_ctY(scal_eqn, scal_vec, sym, sh, m, Lmax, pol, tor):
	#----------------------------------------------------------------
	""" Compute the spherical harmonic coefficients of the projections cos(theta) Ylm and sin(theta) * dYlm/dtheta onto Y_{l-1}^m and Y_{l+1}^m.
	Inputs :
		* scal_eqn : scalar of the equation ['pol', 'tor'],
		* scal_vec : scalar of the (eigen)vector ['pol', 'tor'],
		* sym      : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* sh       : python instance of SHTns.
		* m        : azimuthal order,
		* Lmax     : truncation of the spherical harmonic degrees,
		* pol      : python instance of the poloidal-like scalar [only spherical degrees matter], in pratice the poloidal one,
		* tor      : python instance of the toroidal-like scalar [only spherical degrees matter], in pratice the toroidal one.
	Outputs :
		* l_p1 : spherical harmonic degrees for the loop on the scalar equation for Y_{l+1}^m,
		* l_m1 : spherical harmonic degrees for the loop on the scalar equation for Y_{l-1}^m,
		* coeff_p1_stdY : spherical harmonic coefficients of sin(theta) * dYlm/dtheta of Y_{l+1}^m for each l in l_p1,
		* coeff_m1_stdY : spherical harmonic coefficients of sin(theta) * dYlm/dtheta of Y_{l-1}^m for each l in l_m1,
		* coeff_p1_ctY  : spherical harmonic coefficients of cos(theta) * Ylm of Y_{l+1}^m for each l in l_p1,
		* coeff_m1_ctY  : spherical harmonic coefficients of cos(theta) * Ylm of Y_{l-1}^m  for each l in l_m1.
	"""
	#----------------------------------------------------------------
	# Initialization
	# \alpha_{l-1}^l = \sqrt{(l^2-m^2)/(4l^2-1)}, \alpha_{l+1}^l = \sqrt{((l+1)^2-m^2)/(4(l+1)^2-1)}
	ct_Y  = sh.mul_ct_matrix()		# \alpha_{l-1}^l, \alpha_{l+1}^l
	st_dY = sh.st_dt_matrix()		# (l-1) * \alpha_{l-1}^l, -(l+2) * \alpha_{l+1}^l
	coeff_p1_stdY, coeff_m1_stdY, coeff_p1_ctY, coeff_m1_ctY = [], [], [], []

	if scal_eqn == 'pol' and scal_vec == 'tor':		# 'Poloidal'-like equation
		# even modes
		if sym == 'pos':
		# P_m, P_{m+2}, ... and T_{m+1}, T_{m+3}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (P_2, T_1) and end at (P_Lmax, T_{Lmax-1})
				l_p1 = np.arange(pol.L[0], pol.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(pol.L[0], pol.L[-1], 2, dtype=int)			# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at (P_m, T_{m+1}) and end at (P_{Lmax-1}, T_Lmax)
				l_p1 = np.arange(pol.L[1], pol.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(pol.L[0], pol.L[-1]+1, 2, dtype=int)		# Y_{l-1}
		# odd modes
		elif sym == 'neg':
		# P_{m+1}, P_{m+3}, ... and T_m, T_{m+2}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (P_1, T_2) and end at (P_{Lmax-1}, T_Lmax)
				l_p1 = np.arange(pol.L[1], pol.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(pol.L[0], pol.L[-1]+1, 2, dtype=int)		# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at P_{m+1} T_m and end at P_Lmax and T_{Lmax-1}
				l_p1 = np.arange(pol.L[0], pol.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(pol.L[0], pol.L[-1], 2, dtype=int)			# Y_{l-1}

	elif scal_eqn == 'tor' and scal_vec == 'pol':	# 'Toroidal'-like equation
		# even modes
		if sym == 'pos':
		# P_m, P_{m+2}, ... and T_{m+1}, T_{m+3}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (P_2, T_1) and end at (P_Lmax, T_{Lmax-1})
				l_p1 = np.arange(tor.L[1], tor.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(tor.L[0], tor.L[-1]+1, 2, dtype=int)		# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at (P_m, T_{m+1}) and end at (P_{Lmax-1}, T_Lmax)
				l_p1 = np.arange(tor.L[0], tor.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(tor.L[0], tor.L[-1], 2, dtype=int)			# Y_{l-1}
		# odd modes
		if sym == 'neg':
		# P_{m+1}, P_{m+3}, ... and T_m, T_{m+2}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (P_1, T_2) and end at (P_{Lmax-1}, T_Lmax)
				l_p1 = np.arange(tor.L[0], tor.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(tor.L[0], tor.L[-1], 2, dtype=int)			# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at P_{m+1} T_m and end at P_Lmax and T_{Lmax-1}
				l_p1 = np.arange(tor.L[1], tor.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(tor.L[0], tor.L[-1]+1, 2, dtype=int)		# Y_{l-1}

	for l in l_p1:
		ind_Ylm_pos = 2*sh.idx(int(l), int(m))
		coeff_p1_stdY.append(st_dY[ind_Ylm_pos])		# (l-1) * \alpha_{l-1}^l
		coeff_p1_ctY.append(ct_Y[ind_Ylm_pos])			# \alpha_{l-1}^2
	for l in l_m1:
		ind_Ylm_neg = 2*sh.idx(int(l), int(m)) + 1
		coeff_m1_stdY.append(st_dY[ind_Ylm_neg])		# -(l+2) * \alpha_{l+1}^l
		coeff_m1_ctY.append(ct_Y[ind_Ylm_neg])			# \alpha_{l+1}^2

	return l_p1, l_m1, np.array(coeff_p1_stdY), np.array(coeff_m1_stdY), np.array(coeff_p1_ctY), np.array(coeff_m1_ctY)

def magnetic_coeff_A(scal_eqn, scal_vec, sym, sh, m, Lmax, pol, tor, polb, torb):
	#----------------------------------------------------------------
	""" Compute the coefficients A_{l-1}^m and A_{l+1}^m.
	Inputs :
		* scal_eqn : scalar of the equation ['pol', 'tor', 'polb', 'torb'],
		* scal_vec : scalar of the (eigen)vector ['pol', 'tor', 'polb', 'torb'],
		* sym      : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* sh       : python instance of SHTns.
		* m        : azimuthal order,
		* Lmax     : truncation of the spherical harmonic degrees,
		* pol      : python instance of the poloidal-like scalar [only spherical degrees matter], in pratice the poloidal one,
		* tor      : python instance of the toroidal-like scalar [only spherical degrees matter], in pratice the toroidal one,
		* polb     : python instance of the poloidal_b-like scalar [only spherical degrees matter], in pratice the poloidal_b one,
		* torb     : python instance of the toroidal_b-like scalar [only spherical degrees matter], in pratice the toroidal_b one.
	Outputs :
		* l_p1 : spherical harmonic degrees for the loop on the scalar equation for Y_{l+1}^m,
		* l_m1 : spherical harmonic degrees for the loop on the scalar equation for Y_{l-1}^m,
		* coeff_p1_A  : coefficients A_{l-1}^m for each l in l_p1,
		* coeff_m1_A  : coefficients A_{l+1}^m for each l in l_m1.
	"""
	#----------------------------------------------------------------
	# Initialization
	# \alpha_{l-1}^l = \sqrt{(l^2-m^2)/(4l^2-1)}, \alpha_{l+1}^l = \sqrt{((l+1)^2-m^2)/(4(l+1)^2-1)}
	ct_Y  = sh.mul_ct_matrix()		# \alpha_{l-1}^l, \alpha_{l+1}^l
	coeff_p1_A, coeff_m1_A = [], []

	if scal_eqn == 'pol' and scal_vec == 'polb':		# 'Poloidal'-like equation
		# even modes
		if sym == 'pos':
		# P_m, P_{m+2}, ... and Pb_{m+1}, Pb_{m+3}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (P_2, Pb_1) and end at (P_Lmax, Pb_{Lmax-1})
				l_p1 = np.arange(pol.L[0], pol.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(pol.L[0], pol.L[-1], 2, dtype=int)			# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at (P_m, Pb_{m+1}) and end at (P_{Lmax-1}, Pb_Lmax)
				l_p1 = np.arange(pol.L[1], pol.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(pol.L[0], pol.L[-1]+1, 2, dtype=int)		# Y_{l-1}
		# odd modes
		elif sym == 'neg':
		# P_{m+1}, P_{m+3}, ... and Pb_m, Pb_{m+2}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (P_1, Pb_2) and end at (P_{Lmax-1}, Pb_Lmax)
				l_p1 = np.arange(pol.L[1], pol.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(pol.L[0], pol.L[-1]+1, 2, dtype=int)		# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at P_{m+1} Pb_m and end at P_Lmax and Pb_{Lmax-1}
				l_p1 = np.arange(pol.L[0], pol.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(pol.L[0], pol.L[-1], 2, dtype=int)			# Y_{l-1}

	elif scal_eqn == 'tor' and scal_vec == 'torb':		# 'Toroidal'-like equation
		# even modes
		if sym == 'pos':
		# Tb_m, Tb_{m+2}, ... and T_{m+1}, T_{m+3}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (Tb_2, T_1) and end at (Tb_Lmax, T_{Lmax-1})
				l_p1 = np.arange(tor.L[1], tor.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(tor.L[0], tor.L[-1]+1, 2, dtype=int)		# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at (Tb_m, T_{m+1}) and end at (Tb_{Lmax-1}, T_Lmax)
				l_p1 = np.arange(tor.L[0], tor.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(tor.L[0], tor.L[-1], 2, dtype=int)			# Y_{l-1}
		# odd modes
		if sym == 'neg':
		# Tb_{m+1}, Tb_{m+3}, ... and T_m, T_{m+2}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (Tb_1, T_2) and end at (Tb_{Lmax-1}, T_Lmax)
				l_p1 = np.arange(tor.L[0], tor.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(tor.L[0], tor.L[-1], 2, dtype=int)			# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at Tb_{m+1} T_m and end at Tb_Lmax and T_{Lmax-1}
				l_p1 = np.arange(tor.L[1], tor.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(tor.L[0], tor.L[-1]+1, 2, dtype=int)		# Y_{l-1}

	elif scal_eqn == 'polb' and scal_vec == 'pol':		# 'Poloidal_b'-like equation
		# even modes
		if sym == 'pos':
		# P_m, P_{m+2}, ... and Pb_{m+1}, Pb_{m+3}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (P_2, Pb_1) and end at (P_Lmax, Pb_{Lmax-1})
				l_p1 = np.arange(polb.L[1], polb.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(polb.L[0], polb.L[-1]+1, 2, dtype=int)		# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at (P_m, Pb_{m+1}) and end at (P_{Lmax-1}, Pb_Lmax)
				l_p1 = np.arange(polb.L[0], polb.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(polb.L[0], polb.L[-1], 2, dtype=int)		# Y_{l-1}
		# odd modes
		elif sym == 'neg':
		# P_{m+1}, P_{m+3}, ... and Pb_m, Pb_{m+2}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (P_1, Pb_2) and end at (P_{Lmax-1}, Pb_Lmax)
				l_p1 = np.arange(polb.L[0], polb.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(polb.L[0], polb.L[-1], 2, dtype=int)		# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at P_{m+1} Pb_m and end at P_Lmax and Pb_{Lmax-1}
				l_p1 = np.arange(polb.L[1], polb.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(polb.L[0], polb.L[-1]+1, 2, dtype=int)		# Y_{l-1}

	elif scal_eqn == 'torb' and scal_vec == 'tor':		# 'Toroidal_b'-like equation
		# even modes
		if sym == 'pos':
		# Tb_m, Tb_{m+2}, ... and T_{m+1}, T_{m+3}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (Tb_2, T_1) and end at (Tb_Lmax, T_{Lmax-1})
				l_p1 = np.arange(torb.L[0], torb.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(torb.L[0], torb.L[-1], 2, dtype=int)		# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at (Tb_m, T_{m+1}) and end at (Tb_{Lmax-1}, T_Lmax)
				l_p1 = np.arange(torb.L[1], torb.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(torb.L[0], torb.L[-1]+1, 2, dtype=int)		# Y_{l-1}
		# odd modes
		elif sym == 'neg':
		# Tb_{m+1}, Tb_{m+3}, ... and T_m, T_{m+2}, ...
			if   Lmax%2 == 0 and m == 0:
			# Harmonics start at (Tb_1, T_2) and end at (Tb_{Lmax-1}, T_Lmax)
				l_p1 = np.arange(torb.L[1], torb.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(torb.L[0], torb.L[-1]+1, 2, dtype=int)		# Y_{l-1}
			elif (Lmax%2 != 0 and m%2 == 0 and m > 0) or (Lmax%2 == 0 and m%2 != 0):
			# Harmonics start at Tb_{m+1} T_m and end at Tb_Lmax and T_{Lmax-1}
				l_p1 = np.arange(torb.L[0], torb.L[-1]+1, 2, dtype=int)		# Y_{l+1}
				l_m1 = np.arange(torb.L[0], torb.L[-1], 2, dtype=int)		# Y_{l-1}

	for l in l_p1:
		ind_Ylm_pos = 2*sh.idx(int(l), int(m))
		coeff_p1_A.append(ct_Y[ind_Ylm_pos]/l**2)			# A_{l-1} = \alpha_{l-1}^2 / l^2
	for l in l_m1:
		ind_Ylm_neg = 2*sh.idx(int(l), int(m)) + 1
		coeff_m1_A.append(ct_Y[ind_Ylm_neg]/(l+1)**2)		# A_{l+1} = \alpha_{l+1}^2 / (l+1)^2

	return l_p1, l_m1, np.array(coeff_p1_A), np.array(coeff_m1_A)

#-------------------------------------------------------------------#
#						Filling of the matrix A						#
#-------------------------------------------------------------------#
def hydro_eig_A(A, grid, sh, m, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi, Omega0, nu, flag_temp):
	#----------------------------------------------------------------
	""" Filling of the inertial blocks of the matrix A of the problem with the eigenvalues.
	Inputs :
		* A         : Scipy sparse matrix,
		* grid      : radial grid class,
		* sh        : class defined with SHTns,
		* m         : azimuthal order [integer],
		* Lmax      : harmonic degree of truncation [even integer],
		* sym       : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* bc_i      : 0 = BC at r = 0; 1 = FS at ICB; 2 = NS at ICB,
		* bc_o      : 1 = FS at CMB; 2 = NS at CMB,
		* bc_i_temp : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = T zero,
		* bc_o_temp : 1 = Flux imposes zero on the CMB; 2 = T zero,
		* Omega0    : angular velocity in the rotating frame,
		* nu        : kinematic viscosity,
		* flag_temp : enable or not the temperature field.
	Output :
		* A         : Scipy sparse matrix.
	"""
	#----------------------------------------------------------------
	#------------------------
	# Initialization
	#------------------------
	pol, tor, temp, chi, polb, torb = nb_eig_vec(m, grid.N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	lapla = sub.mat_lapla(grid)
	d1 = sub.mat_deriv1(grid)

	#------------------------
	# Block poloidal (eqn) - poloidal (scal) : Laplacian and Bilaplacian
	#------------------------
	for l in pol.L:
		# Non-zero entries
		M_PolPol = lapla - l*(l+1) * np.diag(1/grid.r[1:-1]**2)					# \Delta_l = d^2/dr^2 + 2/r*d/dr - l*(l+1)/r^2
		M_PolPol = 2*Omega0 * 1j*m/(l*(l+1)) * M_PolPol							# 2*Omega0*i*m/(l*(l+1)) * \Delta_l
		if nu != 0:
			#Delta_l2 = nu * sub.mat_bilapla_pol(lapla, grid, l, bc_i, bc_o)		# nu * \Delta_l\Delta_l
			Delta_l2 = nu * sub.mat_bilapla_pol(grid, l, bc_i, bc_o)			# nu * \Delta_l\Delta_l
			M_PolPol += Delta_l2
			del Delta_l2
		i_PolPol, j_PolPol = np.nonzero(M_PolPol)
		I, J = ind_eig_mat('pol', 'pol', l, l, i_PolPol, j_PolPol, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolPol[i_PolPol[k], j_PolPol[k]]
		del M_PolPol, i_PolPol, j_PolPol, I, J

	#------------------------
	# Block poloidal (eqn) - toroidal (scal)
	# Attention to the coupling L_1 and L_{-1} which depends on pos/ned and m/Lmax
	# N-1 eqn for N-1 a N+1 columns
	# Loops on the equations to simplify the coefficients
	#------------------------
	l_p1, l_m1, coeff_p1_stdY, coeff_m1_stdY, coeff_p1_ctY, coeff_m1_ctY = coeff_stdY_ctY('pol', 'tor', sym, sh, m, Lmax, pol, tor)
	# L_1 operator
	for ind in range(len(l_p1)):
		# Non-zero entries
		l = l_p1[ind]
		M_PolTor = 2*Omega0/(l*(l+1)) * sub.mat_L_tor(d1, grid, l-1, 1, coeff_p1_stdY[ind], coeff_p1_ctY[ind], bc_i, bc_o)
		i_PolTor, j_PolTor = np.nonzero(M_PolTor)
		I, J = ind_eig_mat('pol', 'tor', l, l-1, i_PolTor, j_PolTor, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolTor[i_PolTor[k], j_PolTor[k]]
		del M_PolTor, i_PolTor, j_PolTor, I, J
	# L_{-1} operator
	for ind in range(len(l_m1)):
		# Non-zero entries
		l = l_m1[ind]
		M_PolTor = 2*Omega0/(l*(l+1)) * sub.mat_L_tor(d1, grid, l+1, -1, coeff_m1_stdY[ind], coeff_m1_ctY[ind], bc_i, bc_o)
		i_PolTor, j_PolTor = np.nonzero(M_PolTor)
		I, J = ind_eig_mat('pol', 'tor', l, l+1, i_PolTor, j_PolTor, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolTor[i_PolTor[k], j_PolTor[k]]
		del M_PolTor, i_PolTor, j_PolTor, I, J

	#------------------------
	# Block toroidal (eqn) - toroidal (scal)
	#------------------------
	for l in tor.L:
		# Initialization
		Id = 2*Omega0 * 1j*m/(l*(l+1)) * np.eye(tor.nbel, dtype=float)			# 2*Omega0*i*m/(l*(l+1))
		if nu != 0:
			Delta_l = nu * sub.mat_lapla_tor(lapla, grid, l, bc_i, bc_o)		# nu * \Delta_l
			M_TorTor = Id + Delta_l
			del Id, Delta_l
		i_TorTor, j_TorTor = np.nonzero(M_TorTor)
		I, J = ind_eig_mat('tor', 'tor', l, l, i_TorTor, j_TorTor, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_TorTor[i_TorTor[k], j_TorTor[k]]
		del M_TorTor, i_TorTor, j_TorTor, I, J

	#------------------------
	# Block toroidal (eqn) - poloidal (scal)
	# Attention to the coupling L_1 and L_{-1} which depends on pos/ned and m/Lmax
	# N-1 eqn for N-1 a N+1 columns
	# Loops on the equations to simplify the coefficients
	#------------------------
	l_p1, l_m1, coeff_p1_stdY, coeff_m1_stdY, coeff_p1_ctY, coeff_m1_ctY = coeff_stdY_ctY('tor', 'pol', sym, sh, m, Lmax, pol, tor)
	# L_1 operator
	for ind in range(len(l_p1)):
		# Non-zero entries
		l = l_p1[ind]
		M_TorPol = -2*Omega0/(l*(l+1)) * sub.mat_L_pol(d1, grid, l-1, 1, coeff_p1_stdY[ind], coeff_p1_ctY[ind], bc_i, bc_o)
		i_TorPol, j_TorPol = np.nonzero(M_TorPol)
		I, J = ind_eig_mat('tor', 'pol', l, l-1, i_TorPol, j_TorPol, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_TorPol[i_TorPol[k], j_TorPol[k]]
		del M_TorPol, i_TorPol, j_TorPol, I, J
	# L_{-1} operator
	for ind in range(len(l_m1)):
		# Non-zero entries
		l = l_m1[ind]
		M_TorPol = -2*Omega0/(l*(l+1)) * sub.mat_L_pol(d1, grid, l+1, -1, coeff_m1_stdY[ind], coeff_m1_ctY[ind], bc_i, bc_o)
		i_TorPol, j_TorPol = np.nonzero(M_TorPol)
		I, J = ind_eig_mat('tor', 'pol', l, l+1, i_TorPol, j_TorPol, flag_temp, pol, tor, temp, chi, polb, torb)# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_TorPol[i_TorPol[k], j_TorPol[k]]
		del M_TorPol, i_TorPol, j_TorPol, I, J

	return A

def temp_eig_A(A, grid, m, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi, kappa, kappac, Ra, Rac, N2, N2c, gravity, flag_temp):
	#----------------------------------------------------------------
	""" Filling of the inertial blocks of the matrix A of the problem with the eigenvalues.
	Inputs :
		* A         : Scipy sparse matrix,
		* grid      : radial grid class,
		* m         : azimuthal order [integer],
		* Lmax      : harmonic degree of truncation [even integer],
		* sym       : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* bc_i      : 0 = BC at r = 0; 1 = FS at ICB; 2 = NS at ICB,
		* bc_o      : 1 = FS at CMB; 2 = NS at CMB,
		* bc_i_temp : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = T zero,
		* bc_o_temp : 1 = Flux imposes zero on the CMB; 2 = T zero,
		* Omega0    : angular velocity in the rotating frame,
		* nu        : kinematic viscosity,
		* flag_temp : enable or not the temperature field.
	Output :
		* A         : Scipy sparse matrix.
	"""
	#----------------------------------------------------------------
	#------------------------
	# Initialization
	#------------------------
	N = grid.N
	pol, tor, temp, chi, polb, torb = nb_eig_vec(m, N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	lapla = sub.mat_lapla(grid)

	#------------------------
	# Block poloidal (eqn) - temperature (scal) to resize because temp.nbel >= pol.nbel
	# L_vec = L_eqn here...
	# N-1 eqn for (N-1) a (N+1) columns
	#------------------------
	if Ra != 0:
		M_PolTemp = -Ra * gravity[:]/grid.r[:] * np.eye(N+1, dtype=complex)
		# BC
		M_PolTemp = np.delete(M_PolTemp, 0, axis=0)			# N*(N+1)
		M_PolTemp = np.delete(M_PolTemp, -1, axis=0)		# (N-1)*(N+1)
		if bc_i_temp == 0 or bc_i_temp == 2:
			M_PolTemp = np.delete(M_PolTemp, 0, axis=1)		# (N-1)*N
		if bc_o_temp == 2:
			M_PolTemp = np.delete(M_PolTemp, -1, axis=1)	# (N-1)* N or N-1
		# Assignment
		i_PolTemp, j_PolTemp = np.nonzero(M_PolTemp)
		for l in temp.L:
			I, J = ind_eig_mat('pol', 'temp', l, l, i_PolTemp, j_PolTemp, flag_temp, pol, tor, temp, chi, polb, torb)
			for k in range(len(I)):
				A[I[k], J[k]] = M_PolTemp[i_PolTemp[k], j_PolTemp[k]]
		del M_PolTemp, i_PolTemp, j_PolTemp, I, J

	#------------------------
	# Block temperature (eqn) - temperature (scal)
	#------------------------
	if kappa != 0:
		for l in temp.L:
			# Initialization
			M_TempTemp = kappa * sub.mat_lapla_temp(lapla, grid, l, bc_i_temp, bc_o_temp)
			i_TempTemp, j_TempTemp = np.nonzero(M_TempTemp)
			I, J = ind_eig_mat('temp', 'temp', l, l, i_TempTemp, j_TempTemp, flag_temp, pol, tor, temp, chi, polb, torb)
			for k in range(len(I)):
				A[I[k], J[k]] = M_TempTemp[i_TempTemp[k], j_TempTemp[k]]
		del M_TempTemp, i_TempTemp, j_TempTemp, I, J
	else:
		print("\t\tBeware: Thermal diffusion disabled!")
	
	#------------------------
	# Block temperature (eqn) - poloidal (scal) to resize because temp.nbel !>= pol.nbel
	# L_vec = L_eqn here...
	# (N-1) a (N+1) eqn for N-1 columns
	#------------------------
	if np.any(N2!=0):
		for l in range(pol.Lmin, temp.L[-1]+1, 2):
			M_TempPol = np.diag(-l*(l+1)*N2[1:-1]/grid.r[1:-1])	# (N-1)*(N-1)
			# BC
			if bc_i_temp == 1:
				M_TempPol = np.vstack([np.zeros((1, N-1), dtype=float), M_TempPol])
			if bc_o_temp == 1:
				M_TempPol = np.vstack([M_TempPol, np.zeros((1, N-1), dtype=float)])
			i_TempPol, j_TempPol = np.nonzero(M_TempPol)
			I, J = ind_eig_mat('temp', 'pol', l, l, i_TempPol, j_TempPol, flag_temp, pol, tor, temp, chi, polb, torb)
			for k in range(len(I)):
				A[I[k], J[k]] = M_TempPol[i_TempPol[k], j_TempPol[k]]
		del M_TempPol, i_TempPol, j_TempPol, I, J
	else:
		print("\t\tBeware: Thermal stratification disabled!")
	
	#------------------------
	# Compositional variables
	#------------------------
	if flag_temp == 2:
		#------------------------
		# Block poloidal (eqn) - chi (scal) to resize because chi.nbel >= pol.nbel
		# L_vec = L_eqn here...
		# N-1 eqn for (N-1) a (N+1) columns
		#------------------------
		if Rac != 0:
			M_PolChi = -Rac * gravity[:]/grid.r[:] * np.eye(N+1, dtype=complex)
			# BC
			M_PolChi = np.delete(M_PolChi, 0, axis=0)			# N*(N+1)
			M_PolChi = np.delete(M_PolChi, -1, axis=0)			# (N-1)*(N+1)
			if bc_i_chi == 0 or bc_i_chi == 2:
				M_PolChi = np.delete(M_PolChi, 0, axis=1)		# (N-1)*N
			if bc_o_chi == 2:
				M_PolChi = np.delete(M_PolChi, -1, axis=1)		# (N-1)* N or N-1
			# Assignment
			i_PolChi, j_PolChi = np.nonzero(M_PolChi)
			for l in chi.L:
				I, J = ind_eig_mat('pol', 'chi', l, l, i_PolChi, j_PolChi, flag_temp, pol, tor, temp, chi, polb, torb)
				for k in range(len(I)):
					A[I[k], J[k]] = M_PolChi[i_PolChi[k], j_PolChi[k]]
			del M_PolChi, i_PolChi, j_PolChi, I, J

		#------------------------
		# Block chi (eqn) - chi (scal)
		#------------------------
		if kappac != 0:
			for l in chi.L:
				# Initialization
				M_ChiChi = kappac * sub.mat_lapla_temp(lapla, grid, l, bc_i_chi, bc_o_chi)
				i_ChiChi, j_ChiChi = np.nonzero(M_ChiChi)
				I, J = ind_eig_mat('chi', 'chi', l, l, i_ChiChi, j_ChiChi, flag_temp, pol, tor, temp, chi, polb, torb)
				for k in range(len(I)):
					A[I[k], J[k]] = M_ChiChi[i_ChiChi[k], j_ChiChi[k]]
			del M_ChiChi, i_ChiChi, j_ChiChi, I, J
		else:
			print("\t\tBeware: Compositional diffusion disabled!")
	
		#------------------------
		# Block chi (eqn) - poloidal (scal) to resize because chi.nbel !>= pol.nbel
		# L_vec = L_eqn here...
		# (N-1) a (N+1) eqn for N-1 columns
		#------------------------
		if np.any(N2c!=0):
			for l in range(pol.Lmin, chi.L[-1]+1, 2):
				M_ChiPol = np.diag(-l*(l+1)*N2c[1:-1]/grid.r[1:-1])	# (N-1)*(N-1)
				# BC
				if bc_i_chi == 1:
					M_ChiPol = np.vstack([np.zeros((1, N-1), dtype=float), M_ChiPol])
				if bc_o_chi == 1:
					M_ChiPol = np.vstack([M_ChiPol, np.zeros((1, N-1), dtype=float)])
				i_ChiPol, j_ChiPol = np.nonzero(M_ChiPol)
				I, J = ind_eig_mat('chi', 'pol', l, l, i_ChiPol, j_ChiPol, flag_temp, pol, tor, temp, chi, polb, torb)
				for k in range(len(I)):
					A[I[k], J[k]] = M_ChiPol[i_ChiPol[k], j_ChiPol[k]]
			del M_ChiPol, i_ChiPol, j_ChiPol, I, J
		else:
			print("\t\tBeware: Chemical stratification disabled!")

	return A

def buoyancy_eig_A(A, grid, m, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi, Ra, Rac, gravity, flag_temp):
	#----------------------------------------------------------------
	""" Filling of the temperature blocks of the matrix A with the eigenvalue problem.
	Inputs :
		* A         : hollow matrix [PETSc],
		* grid      : radial grid class,
		* m         : azimuthal order [integer],
		* Lmax      : harmonic degree of truncation [even integer],
		* sym       : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* bc_i      : 0 = BC at r = 0; 1 = FS at ICB; 2 = NS at ICB,
		* bc_o      : 1 = FS at CMB; 2 = NS at CMB,
		* bc_i_temp : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = T zero,
		* bc_o_temp : 1 = Flux imposes zero on the CMB; 2 = T zero,
		* bc_i_chi  : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = zero chi,
		* bc_o_chi  : 1 = Flux imposes zero on the CMB; 2 = zero chi,
		* Ra        : Rayleigh number,
		* N2        : stratified profile [N+1 points],
		* flag_temp : enable or not the temperature field.
	Output :
		* A         : scipy sparse matrix.
	"""
	#----------------------------------------------------------------
	#------------------------
	# Initialization
	#------------------------
	N = grid.N
	pol, tor, temp, chi, polb, torb = nb_eig_vec(m, N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	lapla = sub.mat_lapla(grid)
	np.seterr(divide='ignore')	# avoid warnings from division by zero in the following:

	#------------------------
	# Block poloidal (eqn) - temperature (scal) to resize because temp.nbel >= pol.nbel
	# L_vec = L_eqn here...
	# N-1 eqn for (N-1) a (N+1) columns
	#------------------------
	M_PolTemp = -Ra * gravity[:]/grid.r[:] * np.eye(N+1, dtype=complex)
	# BC
	M_PolTemp = np.delete(M_PolTemp, 0, axis=0)			# N*(N+1)
	M_PolTemp = np.delete(M_PolTemp, -1, axis=0)		# (N-1)*(N+1)
	if bc_i_temp == 0 or bc_i_temp == 2:
		M_PolTemp = np.delete(M_PolTemp, 0, axis=1)		# (N-1)*N
	if bc_o_temp == 2:
		M_PolTemp = np.delete(M_PolTemp, -1, axis=1)	# (N-1)* N or N-1
	# Assignment
	i_PolTemp, j_PolTemp = np.nonzero(M_PolTemp)
	for l in temp.L:
		I, J = ind_eig_mat('pol', 'temp', l, l, i_PolTemp, j_PolTemp, flag_temp, pol, tor, temp, chi, polb, torb)
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolTemp[i_PolTemp[k], j_PolTemp[k]]
	del M_PolTemp, i_PolTemp, j_PolTemp, I, J

	#------------------------
	# Block poloidal (eqn) - chi (scal) to resize because chi.nbel >= pol.nbel
	# L_vec = L_eqn here...
	# N-1 eqn for (N-1) a (N+1) columns
	#------------------------
	if flag_temp == 2:
		M_PolChi = -Rac * gravity[:]/grid.r[:] * np.eye(N+1, dtype=complex)
		# BC
		M_PolChi = np.delete(M_PolChi, 0, axis=0)			# N*(N+1)
		M_PolChi = np.delete(M_PolChi, -1, axis=0)			# (N-1)*(N+1)
		if bc_i_chi == 0 or bc_i_chi == 2:
			M_PolChi = np.delete(M_PolChi, 0, axis=1)		# (N-1)*N
		if bc_o_chi == 2:
			M_PolChi = np.delete(M_PolChi, -1, axis=1)		# (N-1)* N or N-1
		# Assignment
		i_PolChi, j_PolChi = np.nonzero(M_PolChi)
		for l in chi.L:
			I, J = ind_eig_mat('pol', 'chi', l, l, i_PolChi, j_PolChi, flag_temp, pol, tor, temp, chi, polb, torb)
			for k in range(len(I)):
				A[I[k], J[k]] = M_PolChi[i_PolChi[k], j_PolChi[k]]
		del M_PolChi, i_PolChi, j_PolChi, I, J

	return A

def magnetic_eig_A(A, grid, Bfield, sh, m, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi, eta, flag_temp):
	#----------------------------------------------------------------
	""" Filling of the magnetic blocks of the matrix A of the problem with the eigenvalues.
	Inputs :
		* A         : Scipy sparse matrix,
		* grid      : radial grid class,
		* Bfield    : imposed magnetic field class,
		* sh        : class defined with SHTns,
		* m         : azimuthal order [integer],
		* Lmax      : harmonic degree of truncation [even integer],
		* sym       : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* bc_i      : 0 = BC at r = 0; 1 = FS at ICB; 2 = NS at ICB,
		* bc_o      : 1 = FS at CMB; 2 = NS at CMB,
		* bc_i_temp : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = T zero,
		* bc_o_temp : 1 = Flux imposes zero on the CMB; 2 = T zero,
		* bc_i_chi  : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = zero chi,
		* bc_o_chi  : 1 = Flux imposes zero on the CMB; 2 = zero chi,
		* eta       : magnetic diffusivity,
		* flag_temp : enable or not the temperature field.
	Output :
		* A         : Scipy sparse matrix.
	"""
	#----------------------------------------------------------------
	#------------------------
	# Initialization
	#------------------------
	pol, tor, temp, chi, polb, torb = nb_eig_vec(m, grid.N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)

	#------------------------
	# Block poloidal (eqn) - poloidal_b (scal)
	# Attention to the coupling L_{l-1}^G and L_{l+1}^G which depends on pos/neg and m/Lmax
	# N-1 eqn for N+1 columns
	# Loops on the equations to simplify the coefficients
	#------------------------
	l_p1, l_m1, coeff_p1_A, coeff_m1_A = magnetic_coeff_A('pol', 'polb', sym, sh, m, Lmax, pol, tor, polb, torb)
	# L_{l-1}^G operator
	for ind in range(len(l_p1)):
		# Non-zero entries
		l = l_p1[ind]
		M_PolPolb = sub.mat_pol_polb(grid, Bfield, l, 1, coeff_p1_A[ind])
		i_PolPolb, j_PolPolb = np.nonzero(M_PolPolb)
		I, J = ind_eig_mat('pol', 'polb', l, l-1, i_PolPolb, j_PolPolb, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolPolb[i_PolPolb[k], j_PolPolb[k]]
		del M_PolPolb, i_PolPolb, j_PolPolb, I, J
	# L_{l+1}^G operator
	for ind in range(len(l_m1)):
		# Non-zero entries
		l = l_m1[ind]
		M_PolPolb = sub.mat_pol_polb(grid, Bfield, l, -1, coeff_m1_A[ind])
		i_PolPolb, j_PolPolb = np.nonzero(M_PolPolb)
		I, J = ind_eig_mat('pol', 'polb', l, l+1, i_PolPolb, j_PolPolb, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolPolb[i_PolPolb[k], j_PolPolb[k]]
		del M_PolPolb, i_PolPolb, j_PolPolb, I, J

	#------------------------
	# Block poloidal (eqn) - toroidal_b (scal)
	#------------------------
	for l in pol.L:
		M_PolTorb = sub.mat_pol_torb(grid, Bfield, l)		# L_l^c
		M_PolTorb = 1j*m/(l*(l+1)) * M_PolTorb				# imL_l^c/(l*(l+1))
		i_PolTorb, j_PolTorb = np.nonzero(M_PolTorb)
		I, J = ind_eig_mat('pol', 'torb', l, l, i_PolTorb, j_PolTorb, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolTorb[i_PolTorb[k], j_PolTorb[k]]
		del M_PolTorb, i_PolTorb, j_PolTorb, I, J

	#------------------------
	# Block toroidal (eqn) - poloidal_b (scal)
	#------------------------
	for l in tor.L:
		M_TorPolb = sub.mat_tor_polb(grid, Bfield, l, bc_i, bc_o)		# L_l^G
		M_TorPolb = 1j*m * M_TorPolb									# imL_l^G
		i_TorPolb, j_TorPolb = np.nonzero(M_TorPolb)
		I, J = ind_eig_mat('tor', 'polb', l, l, i_TorPolb, j_TorPolb, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_TorPolb[i_TorPolb[k], j_TorPolb[k]]
		del M_TorPolb, i_TorPolb, j_TorPolb, I, J

	#------------------------
	# Block toroidal (eqn) - toroidal_b (scal)
	# Attention to the coupling L_{l-1}^c and L_{l+1}^c which depends on pos/neg and m/Lmax
	# N+1 eqn for N-1 columns
	# Loops on the equations to simplify the coefficients
	#------------------------
	l_p1, l_m1, coeff_p1_A, coeff_m1_A = magnetic_coeff_A('tor', 'torb', sym, sh, m, Lmax, pol, tor, polb, torb)
	# L_{l-1}^c operator
	for ind in range(len(l_p1)):
		# Non-zero entries
		l = l_p1[ind]
		M_TorTorb = sub.mat_tor_torb(grid, Bfield, l, 1, coeff_p1_A[ind], bc_i, bc_o)
		i_TorTorb, j_TorTorb = np.nonzero(M_TorTorb)
		I, J = ind_eig_mat('tor', 'torb', l, l-1, i_TorTorb, j_TorTorb, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_TorTorb[i_TorTorb[k], j_TorTorb[k]]
		del M_TorTorb, i_TorTorb, j_TorTorb, I, J
	# L_{l+1}^c operator
	for ind in range(len(l_m1)):
		# Non-zero entries
		l = l_m1[ind]
		M_TorTorb = sub.mat_tor_torb(grid, Bfield, l, -1, coeff_m1_A[ind], bc_i, bc_o)
		i_TorTorb, j_TorTorb = np.nonzero(M_TorTorb)
		I, J = ind_eig_mat('tor', 'torb', l, l+1, i_TorTorb, j_TorTorb, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_TorTorb[i_TorTorb[k], j_TorTorb[k]]
		del M_TorTorb, i_TorTorb, j_TorTorb, I, J

	#------------------------
	# Block poloidal_b (eqn) - poloidal (scal)
	# Attention to the coupling L_{l-1}^P and L_{l+1}^P which depends on pos/neg and m/Lmax
	# N+1 eqn for N-1 columns
	# Loops on the equations to simplify the coefficients
	#------------------------
	l_p1, l_m1, coeff_p1_A, coeff_m1_A = magnetic_coeff_A('polb', 'pol', sym, sh, m, Lmax, pol, tor, polb, torb)
	# L_{l-1}^P operator
	for ind in range(len(l_p1)):
		# Non-zero entries
		l = l_p1[ind]
		M_PolbPol = sub.mat_polb_pol(grid, Bfield, l, 1, coeff_p1_A[ind])
		i_PolbPol, j_PolbPol = np.nonzero(M_PolbPol)
		I, J = ind_eig_mat('polb', 'pol', l, l-1, i_PolbPol, j_PolbPol, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolbPol[i_PolbPol[k], j_PolbPol[k]]
		del M_PolbPol, i_PolbPol, j_PolbPol, I, J
	# L_{l+1}^P operator
	for ind in range(len(l_m1)):
		# Non-zero entries
		l = l_m1[ind]
		M_PolbPol = sub.mat_polb_pol(grid, Bfield, l, -1, coeff_m1_A[ind])
		i_PolbPol, j_PolbPol = np.nonzero(M_PolbPol)
		I, J = ind_eig_mat('polb', 'pol', l, l+1, i_PolbPol, j_PolbPol, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolbPol[i_PolbPol[k], j_PolbPol[k]]
		del M_PolbPol, i_PolbPol, j_PolbPol, I, J

	#------------------------
	# Block poloidal_b (eqn) - toroidal (scal)
	#------------------------
	for l in polb.L:
		M_PolbTor = 1j*m/(l*(l+1)) * np.diag(Bfield.Br)
		#BC to ICB
		if bc_i == 0 or bc_i == 2:
			M_PolbTor = np.delete(M_PolbTor, 0, axis=1)		# (N+1)*N
		if bc_o == 2:
			M_PolbTor = np.delete(M_PolbTor, -1, axis=1)	# (N+1)*N or (N+1)*(N-1)
		i_PolbTor, j_PolbTor = np.nonzero(M_PolbTor)
		I, J = ind_eig_mat('polb', 'tor', l, l, i_PolbTor, j_PolbTor, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolbTor[i_PolbTor[k], j_PolbTor[k]]
		del M_PolbTor, i_PolbTor, j_PolbTor, I, J

	#------------------------
	# Block poloidal_b (eqn) - poloidal_b (scal)
	#------------------------
	for l in polb.L:
		# Non-zeros entries
		if eta != 0:
			Delta_l = sub.mat_polb_polb(grid, l)		# \Delta_l = d^2/dr^2 + 2/r*d/dr - l*(l+1)/r^2
			M_PolbPolb = eta * Delta_l
			del Delta_l
		i_PolbPolb, j_PolbPolb = np.nonzero(M_PolbPolb)
		I, J = ind_eig_mat('polb', 'polb', l, l, i_PolbPolb, j_PolbPolb, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_PolbPolb[i_PolbPolb[k], j_PolbPolb[k]]
		del M_PolbPolb, i_PolbPolb, j_PolbPolb, I, J

	#------------------------
	# Block toroidal_b (eqn) - poloidal (scal)
	#------------------------
	for l in torb.L:
		M_TorbPol = 1j*m/(l*(l+1)) * sub.mat_torb_pol(grid, Bfield, l)
		i_TorbPol, j_TorbPol = np.nonzero(M_TorbPol)
		I, J = ind_eig_mat('torb', 'pol', l, l, i_TorbPol, j_TorbPol, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_TorbPol[i_TorbPol[k], j_TorbPol[k]]
		del M_TorbPol, i_TorbPol, j_TorbPol, I, J

	#------------------------
	# Block toroidal_b (eqn) - toroidal (scal)
	# Attention to the coupling L_{l-1}^w and L_{l+1}^w which depends on pos/neg and m/Lmax
	# N-1 eqn for (N-1) a (N+1) columns
	# Loops on the equations to simplify the coefficients
	#------------------------
	l_p1, l_m1, coeff_p1_A, coeff_m1_A = magnetic_coeff_A('torb', 'tor', sym, sh, m, Lmax, pol, tor, polb, torb)
	# L_{l-1}^w operator
	for ind in range(len(l_p1)):
		# Non-zero entries
		l = l_p1[ind]
		M_TorbTor = sub.mat_torb_tor(grid, Bfield, l, 1, coeff_p1_A[ind], bc_i, bc_o)
		i_TorbTor, j_TorbTor = np.nonzero(M_TorbTor)
		I, J = ind_eig_mat('torb', 'tor', l, l-1, i_TorbTor, j_TorbTor, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_TorbTor[i_TorbTor[k], j_TorbTor[k]]
		del M_TorbTor, i_TorbTor, j_TorbTor, I, J
	# L_{l+1}^w operator
	for ind in range(len(l_m1)):
		# Non-zero entries
		l = l_m1[ind]
		M_TorbTor = sub.mat_torb_tor(grid, Bfield, l, -1, coeff_m1_A[ind], bc_i, bc_o)
		i_TorbTor, j_TorbTor = np.nonzero(M_TorbTor)
		I, J = ind_eig_mat('torb', 'tor', l, l+1, i_TorbTor, j_TorbTor, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_TorbTor[i_TorbTor[k], j_TorbTor[k]]
		del M_TorbTor, i_TorbTor, j_TorbTor, I, J

	#------------------------
	# Block toroidal_b (eqn) - toroidal_b (scal)
	#------------------------
	for l in torb.L:
		# Non-zeros entries
		if eta != 0:
			Delta_l = sub.mat_torb_torb(grid, l)		# \Delta_l = d^2/dr^2 + 2/r*d/dr - l*(l+1)/r^2
			M_TorbTorb = eta * Delta_l
			del Delta_l
		i_TorbTorb, j_TorbTorb = np.nonzero(M_TorbTorb)
		I, J = ind_eig_mat('torb', 'torb', l, l, i_TorbTorb, j_TorbTorb, flag_temp, pol, tor, temp, chi, polb, torb)
		# Assignment
		for k in range(len(I)):
			A[I[k], J[k]] = M_TorbTorb[i_TorbTorb[k], j_TorbTorb[k]]
		del M_TorbTorb, i_TorbTorb, j_TorbTorb, I, J

	return A

#-------------------------------------------------------------------#
#							Filling of B							#
#-------------------------------------------------------------------#
def hydro_eig_B(B, grid, m, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi, flag_temp):
	#----------------------------------------------------------------
	""" Filling of the matrix B of the problem with the eigenvalues.
	Inputs :
		* B         : Scipy sparse matrix,
		* grid      : radial grid class,
		* m         : azimuthal order [integer],
		* Lmax      : harmonic degree of truncation [even integer],
		* sym       : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* bc_i      : 0 = BC at r = 0; 1 = FS at ICB; 2 = NS at ICB,
		* bc_o      : 1 = FS at CMB; 2 = NS at CMB,
		* bc_i_temp : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = T zero,
		* bc_o_temp : 1 = Flux imposes zero on the CMB; 2 = T zero,
		* bc_i_chi  : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = zero chi,
		* bc_o_chi  : 1 = Flux imposes zero on the CMB; 2 = zero chi,
		* flag_temp : enable or not the temperature field.
	Output :
		* B         : Scipy sparse matrix.
	"""
	#----------------------------------------------------------------
	#------------------------
	# Initialization
	#------------------------
	pol, tor, temp, chi, polb, torb = nb_eig_vec(m, grid.N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)
	lapla = sub.mat_lapla(grid)

	#------------------------
	# Block poloidal (eqn) - poloidal (scal)
	#------------------------
	for l in pol.L:
		# Non-zero entries
		M_PolPol = lapla - l*(l+1) * np.diag(1/grid.r[1:-1]**2)		# \Delta_l = d^2/dr^2 + 2/r*d/dr - l*(l+1)/r^2
		i_PolPol, j_PolPol = np.nonzero(M_PolPol)
		I, J = ind_eig_mat('pol', 'pol', l, l, i_PolPol, j_PolPol, flag_temp, pol, tor, temp, chi, polb, torb)
		for k in range(len(I)):
			B[I[k], J[k]] = M_PolPol[i_PolPol[k], j_PolPol[k]]
	del M_PolPol, i_PolPol, j_PolPol, I, J

	#------------------------
	# Block toroidal (eqn) - toroidal (scal)
	#------------------------
	M_TorTor = np.eye(tor.nbel, dtype=complex)
	i_TorTor, j_TorTor = np.nonzero(M_TorTor)
	for l in tor.L:
		I, J = ind_eig_mat('tor', 'tor', l, l, i_TorTor, j_TorTor, flag_temp, pol, tor, temp, chi, polb, torb)
		for k in range(len(I)):
			B[I[k], J[k]] = M_TorTor[i_TorTor[k], j_TorTor[k]]
	del M_TorTor, i_TorTor, j_TorTor, I, J

	return B

def temp_eig_B(B, grid, m, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi, flag_temp):
	#----------------------------------------------------------------
	"""Filling of the temperature block of the matrix B of the problem with the eigenvalues.
	Inputs :
		* B         : PETSc sparse matrix,
		* grid      : radial grid class,
		* m         : azimuthal order [integer],
		* Lmax      : harmonic degree of truncation [even integer],
		* sym       : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* bc_i      : 0 = BC at r = 0; 1 = FS at ICB; 2 = NS at ICB,
		* bc_o      : 1 = FS at CMB; 2 = NS at CMB,
		* bc_i_temp : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = T zero,
		* bc_o_temp : 1 = Flux imposes zero on the CMB; 2 = T zero,
		* bc_i_chi  : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = zero chi,
		* bc_o_chi  : 1 = Flux imposes zero on the CMB; 2 = zero chi,
		* flag_temp : enable or not the temperature field.
	Output : 
		* B         : scipy sparse matrix.
	"""
	#----------------------------------------------------------------
	#------------------------
	# Initialization
	#------------------------
	pol, tor, temp, chi, polb, torb = nb_eig_vec(m, grid.N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)

	#------------------------
	# Block temperature (eqn) - temperature (scal)
	#------------------------
	M_TempTemp = np.eye(temp.nbel, dtype=float)
	i_TempTemp, j_TempTemp = np.nonzero(M_TempTemp)
	for l in temp.L:
		I, J = ind_eig_mat('temp', 'temp', l, l, i_TempTemp, j_TempTemp, flag_temp, pol, tor, temp, chi, polb, torb)
		for k in range(len(I)):
			B[I[k], J[k]] = M_TempTemp[i_TempTemp[k], j_TempTemp[k]]
	del M_TempTemp, i_TempTemp, j_TempTemp, I, J

	#------------------------
	# Block chi (eqn) - chi (scal)
	#------------------------
	if flag_temp == 2:
		M_ChiChi = np.eye(chi.nbel, dtype=float)
		i_ChiChi, j_ChiChi = np.nonzero(M_ChiChi)
		for l in chi.L:
			I, J = ind_eig_mat('chi', 'chi', l, l, i_ChiChi, j_ChiChi, flag_temp, pol, tor, temp, chi, polb, torb)
			for k in range(len(I)):
				B[I[k], J[k]] = M_ChiChi[i_ChiChi[k], j_ChiChi[k]]
		del M_ChiChi, i_ChiChi, j_ChiChi, I, J

	return B

def magnetic_eig_B(B, grid, m, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi, flag_temp):
	#----------------------------------------------------------------
	"""Filling of the magnetic block of the matrix B of the problem with the eigenvalues.
	Inputs :
		* B         : PETSc sparse matrix,
		* grid      : radial grid class,
		* m         : azimuthal order [integer],
		* Lmax      : harmonic degree of truncation [even integer],
		* sym       : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* bc_i      : 0 = BC at r = 0; 1 = FS at ICB; 2 = NS at ICB,
		* bc_o      : 1 = FS at CMB; 2 = NS at CMB,
		* bc_i_temp : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = T zero,
		* bc_o_temp : 1 = Flux imposes zero on the CMB; 2 = T zero,
		* bc_i_chi  : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = zero chi,
		* bc_o_chi  : 1 = Flux imposes zero on the CMB; 2 = zero chi,
		* flag_temp : enable or not the temperature field.
	Output : 
		* B         : scipy sparse matrix.
	"""
	#----------------------------------------------------------------
	#------------------------
	# Initialization
	#------------------------
	pol, tor, temp, chi, polb, torb = nb_eig_vec(m, grid.N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)

	#------------------------
	# Block poloidal_b (eqn) - poloidal_b (scal)
	#------------------------
	M_PolbPolb = np.eye(polb.nbel, dtype=float)
	i_PolbPolb, j_PolbPolb = np.nonzero(M_PolbPolb)
	for l in polb.L:
		I, J = ind_eig_mat('polb', 'polb', l, l, i_PolbPolb, j_PolbPolb, flag_temp, pol, tor, temp, chi, polb, torb)
		for k in range(len(I)):
			B[I[k], J[k]] = M_PolbPolb[i_PolbPolb[k], j_PolbPolb[k]]
	del M_PolbPolb, i_PolbPolb, j_PolbPolb, I, J

	#------------------------
	# Block toroidal_b (eqn) - toroidal_b (scal)
	#------------------------
	M_TorbTorb = np.eye(torb.nbel, dtype=float)
	i_TorbTorb, j_TorbTorb = np.nonzero(M_TorbTorb)
	for l in torb.L:
		I, J = ind_eig_mat('torb', 'torb', l, l, i_TorbTorb, j_TorbTorb, flag_temp, pol, tor, temp, chi, polb, torb)
		for k in range(len(I)):
			B[I[k], J[k]] = M_TorbTorb[i_TorbTorb[k], j_TorbTorb[k]]
	del M_TorbTorb, i_TorbTorb, j_TorbTorb, I, J

	return B

#-------------------------------------------------------------------#
#			Sub functions for magsinge_to_xshells.py				#
#-------------------------------------------------------------------#
def choice_eig(ind_vec, scal, sol, flag_temp, m, N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi):
	#-------------------------------------------------------------------
	""" Choice of the eigenvalue then then formatting of the eigenvector.
	Inputs :
		* ind_vec   : eigenvalue index,
		* scal      : scalar [pol, tor, temp, chi, polb, torb],
		* sol       : matrix of eigenvectors in columns,
		* flag_temp : enable or not the temperature field.
		* m         : azimuthal order of the mode,
		* N         : nb of shelf intervals,
		* Lmax      : harmonic degree of truncation [even integer],
		* sym       : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* bc_i      : 0 = BC at r = 0; 1 = FS at ICB; 2 = NS at ICB,
		* bc_o      : 1 = FS at CMB; 2 = NS at CMB,
		* bc_i_temp : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = T zero,
		* bc_o_temp : 1 = Flux imposes zero on the CMB; 2 = T zero,
		* bc_i_chi  : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = zero chi,
		* bc_o_chi  : 1 = Flux imposes zero on the CMB; 2 = zero chi,
	Output :
		* vec       : eigenvector for the scalar.
	"""
	#-------------------------------------------------------------------
	def _choice_eig(scal, scalv):
		vec = np.zeros((scalv.nbel, scalv.nbl), dtype=complex)
		ind_r = np.arange(scalv.nbel)
		for ind_l in range(scalv.nbl):
			I = ind_eig_vec(scal, scalv.L[ind_l], ind_r, flag_temp, pol, tor, temp, chi, polb, torb)
			vec[ind_r, ind_l] = sol.vec[ind_vec, I]
		return vec

	# Import of constants
	pol, tor, temp, chi, polb, torb = nb_eig_vec(m, N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)

	# Choice of scalar
	if scal == 'pol':
		scalv = pol
	elif scal == 'tor':
		scalv = tor
	elif scal == 'temp':
		scalv = temp
	elif scal == 'chi':
		scalv = chi
	elif scal == 'polb':
		scalv = polb
	elif scal == 'torb':
		scalv = torb

	vec = _choice_eig(scal, scalv)

	return vec

def scal_shtns(scal, vec, m, N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi):
	#-------------------------------------------------------------------
	"""Reshape for shtns
	Inputs :
		* scal      : scalar [pol, tor, temp, chi],
		* vec       : eigenvector given by choice_eig,
		* m         : azimuthal wave number,
		* N         : nb of radial shells,
		* Lmax      : harmonic degree of truncation [even integer],
		* sym       : equatorial symmetry. 'pos':equatorially symmetric (e.g. QG). 'neg': equatorially anti-symmetric,
		* bc_i      : 0 = BC at r = 0; 1 = FS at ICB; 2 = NS at ICB,
		* bc_o      : 1 = FS at CMB; 2 = NS at CMB,
		* bc_i_temp : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = T zero,
		* bc_o_temp : 1 = Flux imposes zero on the CMB; 2 = T zero,
		* bc_i_chi  : 0 = BC at r = 0; 1 = Flux imposes zero on the ICB; 2 = zero chi,
		* bc_o_chi  : 1 = Flux imposes zero on the CMB; 2 = zero chi.
	Output :
		* vec : eigenvector in shtns format.
	"""
	#-------------------------------------------------------------------
	def _scal_shtns(vec, m, Lmax, pol):
		L = np.arange(m, Lmax+1)

		if m == 0:
			vec_shtns = np.zeros((pol.nbel, 2*pol.nbl+1), dtype=complex)
			j_ini = np.where(pol.L[0] == L)[0][0]
			j_end = np.where(pol.L[-1] == L)[0][0]
			k = 0
			for j in range(j_ini, j_end+1, 2):
				vec_shtns[:, j] = vec[:, k]
				k += 1
		elif m > 0:
			vec_shtns = np.zeros((pol.nbel, 2*pol.nbl), dtype=complex)
			j_ini = np.where(pol.L[0] == L)[0][0]
			j_end = np.where(pol.L[-1] == L)[0][0]
			k = 0
			for j in range(j_ini, j_end+1, 2):
				vec_shtns[:, j] = vec[:, k]
				k += 1
			# Concatenation of the zero coefs for l = 0:Lmax and m = 0 (in all cases)
			vec_shtns = np.hstack([np.zeros((pol.nbel, Lmax+1), dtype=complex), vec_shtns])

		return vec_shtns

	# Import of constants
	pol, tor, temp, chi, polb, torb = nb_eig_vec(m, N, Lmax, sym, bc_i, bc_o, bc_i_temp, bc_o_temp, bc_i_chi, bc_o_chi)

	# Choice of scalar
	if scal == 'pol':
		vec_shtns = _scal_shtns(vec, m, Lmax, pol)
	elif scal == 'tor':
		vec_shtns = _scal_shtns(vec, m, Lmax, tor)
	elif scal == 'temp':
		vec_shtns = _scal_shtns(vec, m, Lmax, temp)
	elif scal == 'chi':
		vec_shtns = _scal_shtns(vec, m, Lmax, chi)
	elif scal == 'polb':
		vec_shtns = _scal_shtns(vec, m, Lmax, polb)
	elif scal == 'torb':
		vec_shtns = _scal_shtns(vec, m, Lmax, torb)

	return vec_shtns