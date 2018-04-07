import os
import numpy as np
from scipy.interpolate import interp1d


def normalize(spec_int,spec_obs):
	
	#normalize modelled spectrum to observed one
	peak_obs = np.min(spec_obs[:, 1])
	peak_mod = np.min(spec_int[1, :])

	spec_int_norm = np.divide(spec_int[1, :], peak_mod)
	spec_int_mod = np.multiply(spec_int_norm, peak_obs)

	return spec_int_mod

#-------------------------------------------------#
#CHI2 and RESIDUAL spectrum                       #
#-------------------------------------------------#
def chi_res(s_mod, s_obs,cfg_par):
    
    VROT = cfg_par['vel_pars'].get('vrot', 200)
    DISP = cfg_par['vel_pars'].get('disp', 200)


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

    GAL = cfg_par['general'].get('index_out', '1')

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


    #open output table
    if os.path.exists(out_table) is True: 
        table_out = open(out_table, "ab+")
    else:
        table_out = open(out_table, "ab+")
        #write title line if table does not exist
        title_line_1 = '''#RUN,GAL,RA,DEC,z,v_sys[kms],D_L[Mpc],PA_cont[degrees],'''
        title_line_2 = '''R_max[pc],R_min[pc],H[pc],I[degrees],PA[degrees]'''
        title_line_3 = '''R_maxD2[pc],R_minD2[pc],H_D2[pc],I_D2[degrees],PA_D2[degrees],'''
        title_line_4 = '''v_rot[kms],sign[-],v_res[kms],'''
        title_line_5 = '''res_in[pc],res_fin[pc],chi_square\n'''

        title_line = title_line_1+title_line_2+title_line_3+title_line_4+title_line_5
        table_out.write(title_line)
     
    RUN = sum(1 for line in table_out)
          
    line_1 = str(RUN)+''','''+str(GAL)+''','''+str(RA)+''','''+str(DEC)+''','''
    line_2 = str(z_red)+''','''+str(VSYS)+''','''+str(D_L)+''','''+str(PA_C)+''','''   
    line_3 = str(RMAX)+''','''+str(RMIN)+''','''+str(H0)+','+str(I)+','+str(PA)+''','''+str(SIGN) 
    line_4 = str(RMAXD2)+''','''+str(RMIND2)+''','''+str(H0D2)+''','''+str(ID2)+''','''+str(PAD2)+''','''+str(SIGND2)  
    line_5 = str(VROT)+''','''+str(2*DISP)+''',''' 
    line_6 = str(RES)+''','''+str(RES_FIN)+'''\n'''

    line = line_1+line_2+line_3+line_4+line_5+line_6
    table_out.write(line) 
    
    table_out.close()

    return 0