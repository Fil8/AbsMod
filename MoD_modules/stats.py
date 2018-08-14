import os, string
import numpy as np
from scipy.interpolate import interp1d


def normalize(spec_int,spec_obs):
	
	#normalize modelled spectrum to observed one
	peak_obs = np.min(spec_obs[:, 1])
	peak_mod = np.min(spec_int[1, :])
	if peak_mod < 0.:
		spec_int_norm = np.divide(spec_int[1, :], peak_mod)
		spec_int_mod = np.multiply(spec_int_norm, peak_obs)

	else:
		spec_int_mod = spec_int[1, :]

	return spec_int_mod

#-------------------------------------------------#
#CHI2 and RESIDUAL spectrum                       #
#-------------------------------------------------#
def chi_res(s_mod, s_obs,cfg_par):
	
	VROT = cfg_par['vel_pars'].get('vrot', 200)
	DISP = cfg_par['vel_pars'].get('disp', 200)

	if cfg_par['general'].get('spectrum_type') == 'real':
		#interpolate to have arrays all of the same length
		func_obs = interp1d(s_obs[:, 0], s_obs[:, 1])
		func_mod = interp1d(s_mod[0, :], s_mod[1, :])
		
		#create a new array long enough
		vel_int = np.arange(-VROT, VROT + DISP, 2. * DISP)

		#determine the fluxes of observed 
		#and modelled spectra
		
		obs_int = func_obs(vel_int)
		mod_int = func_mod(vel_int)

		#determine residuals array
		res = obs_int-mod_int

		#set arrays for output
		res_out = np.array([vel_int, res])
		obs_out = np.array([vel_int, obs_int])
		mod_out = np.array([vel_int, mod_int])
		
		noise = np.std(obs_out[1, :])
		inv_noise = 1./(noise*noise)
		
		chi_square = -0.5*np.sum((np.power(res_out[1, :], 2)*inv_noise-np.log(inv_noise)))
	
	else:
		chi_square = 0.0
		func_mod = interp1d(s_mod[0, :], s_mod[1, :])
		vel_int = np.arange(-VROT, VROT + DISP, 2. * DISP)
		mod_int = func_mod(vel_int)
		mod_out = np.array([vel_int, mod_int])

		res_out = np.zeros([2,len(vel_int)])
		res_out[0,:] = vel_int
		res_out[1,:] = 0.0
 		obs_out = np.zeros([2, len(vel_int)])	
 		obs_out[0,:] = vel_int
 		obs_out[1,:] = 0.0	

	
	return res_out, obs_out, mod_out, chi_square

# -------------------------------------------------#
# FWHM & FW20                                      #
# -------------------------------------------------#
def widths(spec_model):
	#find peak of the line
	array = spec_model[1, :].copy()
	array_vels = spec_model[0, :].copy()
	
	if sum(array) != 0.0:
		minimum = np.min(array)
		minimum_idx = np.where(np.abs(array - minimum) == np.abs(array - minimum).min())[0]
	else:
		minimum_idx = 10
		minimum = 25
	#FWHM
	left_array = array[0: int(minimum_idx)]
	right_array = array[int(minimum_idx): len(array)]

	fwhm_min = minimum/2. 
	
	left_fwhm_idx = np.max(np.where(np.abs(left_array - fwhm_min) == 
							 np.abs(left_array - fwhm_min).min())[0])   
	right_fwhm_idx = np.min(np.where(np.abs(right_array - fwhm_min) == 
							  np.abs(right_array - fwhm_min).min())[0])
	
	vel_left = array_vels[left_fwhm_idx]
	vel_right = array_vels[int(minimum_idx+right_fwhm_idx)]
	
	fwhm = np.abs(vel_right - vel_left)
	
	#FW20
	fw20_min = minimum/5. 
	
	left_fw20_idx = np.max(np.where(np.abs(left_array - fw20_min) == 
							 np.abs(left_array - fw20_min).min())[0])  
	right_fw20_idx = np.min(np.where(np.abs(right_array - fw20_min) == 
							  np.abs(right_array - fw20_min).min())[0])

	vel_left = array_vels[left_fw20_idx]
	vel_right = array_vels[int(minimum_idx+right_fw20_idx)]
	
	fw20 = np.abs(vel_right - vel_left)    
	
	return fwhm, fw20


#-------------------------------------------------#
#Write the output table                           #
#-------------------------------------------------#
def write_table(out_table, cfg_par):

	GAL = cfg_par['general'].get('model_name', '1')

	key = 'disk_1'
	RMAX = cfg_par[key].get('rmax', 1000)
	RMIN = cfg_par[key].get('rmin', 0)
	H0 = cfg_par[key].get('h0', 200)
	I = cfg_par[key].get('i', 90)
	PA = cfg_par[key].get('pa', 0)
	SIGN = cfg_par[key].get('sign', 1)
	key = 'disk_2'
	RMAXD2 = cfg_par[key].get('rmax', 1000)
	RMIND2 = cfg_par[key].get('rmin', 0)
	H0D2 = cfg_par[key].get('h0', 200)
	ID2 = cfg_par[key].get('i', 90)
	PAD2 = cfg_par[key].get('pa', 0)
	SIGND2 = cfg_par[key].get('sign', 1)
	
	VROT = cfg_par['vel_pars'].get('vrot', 200)
	DISP = cfg_par['vel_pars'].get('disp', 10)

	PA_C = cfg_par['abs_pars'].get('pa_cont', 200)

	D_L = cfg_par['cont_pars'].get('d_l', 60.)
	z_red = cfg_par['cont_pars'].get('redshift', 0.014)
	VSYS = cfg_par['cont_pars'].get('systemic_vel', 1000.)

	RA = cfg_par['cont_pars'].get('ra', '00:00:00')
	DEC = cfg_par['cont_pars'].get('dec', '00:00:00')

	RES = cfg_par['res_pars'].get('pix_res', 100)
	RES_FIN = cfg_par['res_pars'].get('pix_res_fin', 50)

	FW20 = cfg_par['line_pars'].get('FW20',np.nan)
	FWHM = cfg_par['line_pars'].get('FWHM',np.nan)

	#open output table
	if os.path.exists(out_table) is True: 
		table_out = open(out_table, "r")
		storagearray = []
		for line in table_out:
			search = string.split(line,',')
			if search[0] == GAL:
				storagearray.append(search[0])
		if  len(storagearray) > 0:       
			RUN = len(storagearray)
		else:
			RUN = 0
	else:
		table_out = open(out_table, "ab+")
		#write title line if table does not exist
		title_line_1 = '''#GAL,RUN,RA,DEC,z,v_sys[kms],D_L[Mpc],PA_cont[degrees],'''
		title_line_2 = '''R_max[pc],R_min[pc],H[pc],I[degrees],PA[degrees]'''
		title_line_3 = '''R_maxD2[pc],R_minD2[pc],H_D2[pc],I_D2[degrees],PA_D2[degrees],'''
		title_line_4 = '''v_rot[kms],sign[-],v_res[kms],'''
		title_line_5 = '''res_in[pc],res_fin[pc],chi_square,'''
		title_line_6 = '''FWHM[kms],FW20[kms]\n'''

		title_line = title_line_1+title_line_2+title_line_3+title_line_4+title_line_5+title_line_6
		table_out.write(title_line)
		table_out.close() 
		RUN = 0
	
	table_out = open(out_table, "ab+")
		  
	line_1 = str(GAL)+''','''+str(RUN)+''','''+str(RA)+''','''+str(DEC)+''','''
	line_2 = str(z_red)+''','''+str(VSYS)+''','''+str(D_L)+''','''+str(PA_C)+''','''   
	line_3 = str(RMAX)+''','''+str(RMIN)+''','''+str(H0)+','+str(I)+','+str(PA)+''','''+str(SIGN) +''','''  
	line_4 = str(RMAXD2)+''','''+str(RMIND2)+''','''+str(H0D2)+''','''+str(ID2)+''','''+str(PAD2)+''','''+str(SIGND2)+''','''
	line_5 = str(VROT)+''','''+str(2*DISP)+''',''' 
	line_6 = str(RES)+''','''+str(RES_FIN)+''','''
	line_7 = str(FWHM)+''','''+str(FW20)+'''\n'''

	line = line_1+line_2+line_3+line_4+line_5+line_6+line_7	
	table_out.write(line) 
	
	table_out.close()

	return RUN

def write_table_mcmc(out_table, cfg_par):

	GAL = cfg_par['general'].get('model_name', '1')

	key = 'disk_1'
	RMAX = cfg_par[key].get('rmax', 1000)
	RMIN = cfg_par[key].get('rmin', 0)
	H0 = cfg_par[key].get('h0', 200)
	SIGN = cfg_par[key].get('sign', 1)
	
	key = 'mcmc_pars'
	I_d = cfg_par[key].get('I_left', 0.)
	I_u = cfg_par[key].get('I_right', 90.)

	PA_d = cfg_par[key].get('PA_left', 0.)
	PA_u = cfg_par[key].get('PA_right', 180.)

	VROT = cfg_par['vel_pars'].get('vrot', 200)
	DISP = cfg_par['vel_pars'].get('disp', 10)

	PA_C = cfg_par['abs_pars'].get('pa_cont', 200)

	D_L = cfg_par['cont_pars'].get('d_l', 60.)
	z_red = cfg_par['cont_pars'].get('redshift', 0.014)
	VSYS = cfg_par['cont_pars'].get('systemic_vel', 1000.)

	RA = cfg_par['cont_pars'].get('ra', '00:00:00')
	DEC = cfg_par['cont_pars'].get('dec', '00:00:00')

	RES = cfg_par['res_pars'].get('pix_res', 100)
	RES_FIN = cfg_par['res_pars'].get('pix_res_fin', 50)

	#open output table
	if os.path.exists(out_table) is True: 
		table_out = open(out_table, "r")
		storagearray = []
		for line in table_out:
			search = string.split(line,',')
			if search[0] == GAL:
				storagearray.append(search[0])
		if  len(storagearray) > 0:       
			RUN = len(storagearray)
		else:
			RUN = 0
	else:
		table_out = open(out_table, "ab+")
		#write title line if table does not exist
		title_line_1 = '''#GAL,RUN,I_d[-],I_u[-],PA_d[-],PA_u[-],'''
		title_line_2 = '''RA,DEC,z,v_sys[kms],D_L[Mpc],PA_cont[degrees],'''
		title_line_3 = '''R_max[pc],R_min[pc],H[pc],'''
		title_line_4 = '''v_rot[kms],sign[-],v_res[kms],'''
		title_line_5 = '''res_in[pc],res_fin[pc]'''

		title_line = title_line_1+title_line_2+title_line_3+title_line_4+title_line_5
		table_out.write(title_line)
		table_out.close() 
		RUN = 0
	
	table_out = open(out_table, "ab+")
		  
	line_1 = str(GAL)+''','''+str(RUN)+''','''+str(I_d)+''','''+str(I_u)+''','''+str(PA_d)+''','''+str(PA_u)+''','''
	line_2 = str(RA)+''','''+str(DEC)+''','''+str(z_red)+''','''+str(VSYS)+''','''+str(D_L)+''','''+str(PA_C)+''','''  
	line_3 = str(RMAX)+''','''+str(RMIN)+''','''+str(H0)+''','''+str(SIGN) +''','''  
	line_4 = str(VROT)+''','''+str(2*DISP)+''',''' 
	line_5 = str(RES)+''','''+str(RES_FIN)+''','''

	line = line_1+line_2+line_3+line_4+line_5
	table_out.write(line) 
	
	table_out.close()

	return RUN