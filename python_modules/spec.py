import numpy as np


HI_hz = 1.42040575177e9
C = 2.99792458E5 

def input_spectrum(cfg_par):

	workdir = cfg_par['general'].get('work_directory', None)
	filespec = cfg_par['general'].get('spectrum_name', None)

	VSYS = cfg_par['cont_pars'].get('systemic_vel', None)


	#load observed spectrum
	spec_obs = np.loadtxt(workdir+filespec)
	#convert in frequency given the systemic velocity
	spec_obs[:, 0] = (HI_hz/spec_obs[:, 0]-1)*C
	spec_obs[:, 0] = spec_obs[:, 0]-VSYS


	vels = np.hstack([1.5*spec_obs[0, 0]-0.5*spec_obs[1, 0],
	                  0.5*(spec_obs[0:-1, 0]+spec_obs[1:, 0]),
	                  1.5*spec_obs[-1, 0]-0.5*spec_obs[-2, 0]])
	if vels[-1]<vels[0]:
	        vels=vels[::-1]
	spec_int = np.zeros([2, len(vels)])
	spec_int[0, :] = vels[:]

	#edges of the velocity array
	VRES = cfg_par['vel_pars'].get('vel_res', None)
	DISP = cfg_par['vel_pars'].get('disp', None)

	print '\tedges of the velocity array [km/s]:\t' + str(vels[-1]), str(vels[0])
	print '\tvelocity resolution [km/s]:\t\t' + str(VRES)
	print '\tconvolved velocity resolution [km/s]:\t' + str(2.*DISP)+'\n'

	return vels, spec_int, spec_obs