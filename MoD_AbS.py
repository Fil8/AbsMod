__author__ = "Filippo Maccagni"
__copyright__ = "Fil8"
__email__ = "filippo.maccagni@gmail.com"

import sys, string, os, time
import numpy as np
import yaml, pyaml
#import json

from astropy.io import fits, ascii
from astropy import units as u
from astropy.table import Table, Column, MaskedColumn

import emcee


# Import MoD_AbS modules
sys.path.append('/Users/maccagni/notebooks/MoD_AbS/'+'/python_modules')
import disk as disk
import spec as spec
import cont as cont
import stats as stats
import mcmc_MoD as mcmcmod
import plot_MoD as plotmod
# file_default = '/Users/maccagni/notebooks/MoD_AbS/MoD_AbS_cfg_default.yml'

print '''
 ___  ___  _____   _____  
|   \/   |/  __  \|  __ \ 
| |\  /| || |  | || |  | |  
| | \/ | || |  | || |  | | 
| |    | || |__| || |__/ | 
|_|    |_| \____/ |_____/ 
           ______   ______
    /\    |  __  \ /  ____|
   /  \   | |__| | | (___ 
  / /\ \  |  __  | \___  \ 
 / ____ \ | |__| | ____) |
/_/    \_\|______/|_____/ ''' 


#-------------------------------------------------#
#Load parameter file                              #
#-------------------------------------------------#
print '\n********************\n'   
print '...Parameter File...\n'

file = sys.argv[1]
if file != None:
	cfg = open(file)
else:
	cfg = open(file_default)

cfg_par = yaml.load(cfg)
print pyaml.dump(cfg_par)

#-------------------------------------------------#
#Set output directories and filenames             #
#-------------------------------------------------#



key = 'general'

#IN
workdir = cfg_par[key].get('work_directory', None)
#contname = cfg_par[key].get('continuum_name', None)
#specname = cfg_par[key].get('spectrum_name', None)

#OUT
root_out = workdir+'outputs/'
try:
	os.stat(root_out)
except:
	os.mkdir(root_out)

GAL = cfg_par[key].get('index_out', '1')

#-------------------------------------------------#
#MAIN MAIN MAIN                                   #
#-------------------------------------------------#

print '\n********************\n'   
print '...Begin...\n'

#start timer
tempoinit = time.time()

#-------------------------------------------------#
#INTEGRATED spectrum                              #
#-------------------------------------------------#
#                                                 #
#input integrated spectrum:                       #
#velocity = bins of desired resolution            #
#define the output integrated spectrum            #
#one bin must include zero (zero not an edge!!!)  #
#                                                 #
#-------------------------------------------------#

print '... set spectrum from observed one ...\n'


vels, spec_int, spec_obs = spec.input_spectrum(cfg_par)

#-------------------------------------------------#
#Set the input continuum cube                     # 
#-------------------------------------------------#

print '... set continuum input cube...\n'


#load the continuum image interpolated at the specified resolution
continuum_image , continuum_cube_z = cont.build_continuum(cfg_par)


print '********************\n'

#-------------------------------------------------#
#Compute absorption                              #
#-------------------------------------------------#


key = 'mcmc_pars'
if cfg_par[key].get('enable', False) == True :

	samples, samp_chain = mcmcmod.run_sim(cfg_par)


	# save run outputs
	samples = sampler.chain[:, :].reshape((-1, DIM_mcmc))
	np.savetxt(out_samples_text, samples)
	samp_chain = np.array(sampler.chain[:, :, :], dtype=float)
	np.save(out_walkers_vec, samp_chain)

elif cfg_par[key].get('enable', False) == False :
	print '...compute absorption for disk 1...\n'
	#compute the spectrum and the cubes
	spec_integral, velocity, disk_front, disk_behind = disk.mod_abs(continuum_cube_z, vels, int(1), cfg_par)

	key = 'disk_2'
	if cfg_par[key].get('enable', False) == True :
		print '...compute absorption for disk 2...\n'
		
		continuum_cube_z2 = continuum_cube_z.copy()

		#compute the second set of lines and cubes and 
		#merge them with the first set
		spec_int_2, velocity_2, disk_front_2, disk_behind_2 = disk.mod_abs(continuum_cube_z2,vels, int(2), cfg_par)
		
		#merge absorbed cubes
		velocity = np.nan_to_num(velocity)
		velocity_2 = np.nan_to_num(velocity_2)
		velocity = np.add(velocity, velocity_2) 
		velocity[velocity == 0.0] = np.nan

		#merge disks in front
		disk_front = np.nan_to_num(disk_front)
		disk_front_2 = np.nan_to_num(disk_front_2)
		disk_front = np.add(disk_front, disk_front_2)    
		disk_front[disk_front == 0.0] = np.nan
			
		#merge disks behind
		disk_behind = np.nan_to_num(disk_behind)
		disk_behind_2 = np.nan_to_num(disk_behind_2)
		disk_behind = np.add(disk_behind, disk_behind_2)    
		disk_behind[disk_behind == 0.0] = np.nan

		#merge spectrum
		spec_integral[:] += spec_int_2[:]


	#set final integrated spectrum
	spec_int[1, :] = spec_integral[:] 

	#-------------------------------------------------#
	#NORMALIZE the SPECTRUM to the OBSERVED one       #
	#-------------------------------------------------#
		
	print '...normalize spectrum...\n'
			
	spec_int_mod = stats.normalize(spec_int,spec_obs)

	spec_full = np.array([vels, spec_int_mod])

	print '********************\n'

	#-------------------------------------------------#
	#Determine residuals & chi2 & widths of the lines #
	#-------------------------------------------------#

	print '...compute some stats...\n'

	residuals, obs_res, mod_res, CHI_SQ = stats.chi_res(spec_full, spec_obs, cfg_par)

	FWHM, FW20 = stats.widths(spec_full)

	#-------------------------------------------------#
	#Write new line in output table                   #
	#-------------------------------------------------#
	print '...write table some stats...\n'
	out_table = root_out+GAL+'_table_out.csv'
	stats.write_table(out_table, cfg_par)


	print '********************\n'

	#-------------------------------------------------#
	# PLOT                                            #
	#-------------------------------------------------#

	print '...Begin Plotting...\n'

	key = 'plot_pars'
	if cfg_par[key].get('enable_figure', False) == True :

		form = cfg_par['plot_pars'].get('format', 'pdf')
		out_figure = root_out+'_sum_plot.'+form

		plotmod.plot_figure(spec_obs, spec_full, residuals, out_figure,
				velocity, continuum_image, disk_front, disk_behind, cfg_par)



	print '\n...End Plotting...\n'

#--------------------------------------------------#
# Plot  MCMC                                       #
#--------------------------------------------------#
key = 'plot_pars'
if cfg_par[key].get('enable_mcmc', False) == True :

	#load data
	samples_text = root_out+GAL+'_samples.txt'
	walkers_array = root_out+GAL+'_walkers.npy'
	samples = np.loadtxt(samples_text, dtype=float)
	#rotate samples to sky coordinates
	samples[:,1] -= 90.
	walkers = np.load(walkers_array)
	walkers[:,:,1] -= 90.

	# output names
	form = cfg_par[key].get('format', 'pdf')
	out_samples_figure = root_out+GAL+'_samples.'+form
	out_walkers_figure = root_out+GAL+'_walkers.'+form
 
	print '...Walkers Plotting...\n'
	plotmod.walkers_plot(walkers, out_walkers_figure)
	
	print '...Corner Plotting...\n'
	plotmod.corner_plot(samples, out_samples_figure)

	print '\n...End Plotting...\n'

print '********************\n'   

tempofin = (time.time()-tempoinit)/60.
tempoinit = tempofin

print "\tTotal time: %f minutes\n" % tempofin  

print '...End... \n'

print '********************'   
print 'NORMAL TERMINATION'
print '********************\n' 

###################################################
# END                                             #
###################################################

