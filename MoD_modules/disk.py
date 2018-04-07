import numpy as np
from scipy.ndimage.interpolation import zoom
 
#-------------------------------------------------#
#CUBE of HI disk                                  #
#-------------------------------------------------#

 

#build the cube which contains my disk based on its (RMAX)
#and on the resolution of the 1st cycle

def main_box(cfg_par):
		
	#sets the edges of box to RMAX+20%
	RMAX = cfg_par['disk_1'].get('rmax', None)
	RES = cfg_par['res_pars'].get('pix_res', None)

	x_los = np.arange(-RMAX*1.2, +RMAX*1.2+RES, RES)
	y_los = np.arange(-RMAX*1.2, +RMAX*1.2+RES, RES)
	z_los = np.arange(-RMAX*1.2, +RMAX*1.2+RES, RES)

	#edges of the cube

	if cfg_par['mcmc_pars'].get('enable', False) == False :
		print '\tedges of the cube [pc]:\t\t\t'+str(x_los[-1]), str(x_los[0])
		print '\tsize of the cube [pixels]:\t\t'+str(len(x_los))+' x '+str(len(y_los))+' x '+str(len(z_los))
		print '\tresolution first cycle [pc]:\t\t'+str(RES)+'\n'

	return x_los,y_los,z_los

#-------------------------------------------------#
#Function for the DISK coordinates and velocity   #
#-------------------------------------------------#


def space(z_cube, y_cube, x_cube, continuum_cube, flag, cfg_par):  
	
		#-------------------------------------------------#
		#takes an input cube in space coordinates and the # 
		#continuum cube returns the cube of velocities    #
		#of the disk and the continuum cube with values   # 
		# only where there is absorption                  #
		#-------------------------------------------------#

		#trigonometric parameters of the disk

		if flag == 1:
			key = 'disk_1'
		elif flag == 2:
			key = 'disk_2'

		RMAX = cfg_par[key].get('rmax', 1000)
		RMIN = cfg_par[key].get('rmin', 0)
		H0 = cfg_par[key].get('h0', 200)
		I = cfg_par[key].get('i', 90)
		i_rad = np.radians(I)
		PA = cfg_par[key].get('pa', 0)
		PA += 90.

		pa_rad = np.radians(PA)
		SIGN = cfg_par[key].get('sign', 1)

		VROT = cfg_par['vel_pars'].get('vrot', 200)

		PA_C = cfg_par['abs_pars'].get('pa_cont', 200)
		PA_C_rad = np.radians(PA_C)

		#Disk
		cos_i = np.cos(i_rad)
		sin_i = np.sin(i_rad)
		cos_pa = np.cos(pa_rad)
		sin_pa = np.sin(pa_rad)

		#convert into disk coordinates
		x = cos_pa*x_cube+sin_pa*y_cube
		y = cos_i*(-sin_pa*x_cube+cos_pa*y_cube)+sin_i*z_cube  
		z = -sin_i*(-sin_pa*x_cube+cos_pa*y_cube)+cos_i*z_cube

		#determine the radius of the disk    
		r = np.sqrt(np.power(x, 2)+np.power(y, 2))
		angle = np.arctan2(y, x)        
		#define the cube of velocities 
		if cfg_par['vel_pars'].get('kind', 'flat') == 'flat': 
			vel = -SIGN*sin_i*np.cos(angle)*VROT
		elif cfg_par['vel_pars'].get('kind', 'flat') == 'rising':
			RVFLAT = cfg_par['vel_pars'].get('r_vflat', 1000.)
			vel = -SIGN*sin_i*np.cos(angle)*(VROT/RVFLAT*r)
			#vel = (VROT/RMAX*r)

		#for plotting: define the disk in front of the continuum
		disk_front = vel.copy()
		index_vel = (continuum_cube == 0.0)
		#if np.all(index_vel) != False:
		vel[index_vel] = -np.inf
		#condition for DISK 1
		idx = ((r > RMAX) | (r < RMIN) | (abs(z) >= H0/2.))
		#set to zero or -999 the flux and velocities outside the disk
		continuum_cube[idx] = 0.0
		vel[idx] = -np.inf
		disk_front[idx] = -999.
		#set to non good values what is behind the continuum
		#for plotting: define the disk behind the continuum
		disk_behind = disk_front.copy()
		
		#determine what is in front and what behind the continuum
		index_abs = (z_cube > (np.tan(PA_C_rad)*x_cube))
		
		#modify the cubes: bad values for what is behind
		vel[index_abs] = -np.inf        
		continuum_cube[index_abs] = 0.0
		disk_front[index_abs] = -999.

		index_abs = (z_cube < (np.tan(PA_C_rad)*x_cube))
		
		disk_behind[index_abs] = -999.
		
		#for plotting: exclude the absorbed section from the disk in front
		disk_ind = ((vel >= -VROT) & (vel <= VROT))
		disk_front[disk_ind] = -999.
		
		return vel, continuum_cube, disk_front, disk_behind


#-------------------------------------------------#
#Function computing absorption for each disk     #  
#-------------------------------------------------#

def mod_abs(continuum_cube_z, vels, flag, cfg_par):
 
	#-------------------------------------------------#
	#Set the input coordinates cube                  #
	#-------------------------------------------------#

	if cfg_par['mcmc_pars'].get('enable', False) == False :
		print '...set disk...\n'

	#create a meshgrid for the cube

	x_los, y_los, z_los = main_box(cfg_par)


	z_cube, y_cube, x_cube = np.meshgrid(z_los, y_los, x_los)


	#-------------------------------------------------#
	# VELOCITY and FLUX of the absorbed disk          #
	#-------------------------------------------------#                    
	
	velocity, flusso, disk_front, disk_behind = space(z_cube, y_cube, x_cube, continuum_cube_z, flag, cfg_par)
	
	#-------------------------------------------------#
	# INTERPOLATE to final RESOLUTION                 #
	#-------------------------------------------------#
	if cfg_par['mcmc_pars'].get('enable', False) == False :
		print '...start interpolation...\n'

	#determine the factor for the interpolation depending on 
	#the specified resolutions
	#sets the edges of box to RMAX+20%
	

	RES_FIN = cfg_par['res_pars'].get('pix_res_fin', 50)
	RES = cfg_par['res_pars'].get('pix_res', 100)
	ORDER = cfg_par['res_pars'].get('order', 1)
	factor = RES/RES_FIN
	
	if cfg_par['mcmc_pars'].get('enable', False) == False :
		print '\tInterpolation of a factor:\t'+str(factor)+'\n'

	#print np.ma.masked_invalid(velocity).sum()
	#increase the order for non-linear interpolation
	vel_zoom = zoom(velocity, factor, order=ORDER)
	flux_zoom = zoom(flusso, factor, order=ORDER)

	if cfg_par['mcmc_pars'].get('enable', False) == False :
		print '...end interpolation...\n'

	#-------------------------------------------------#
	# BIN the CUBE in the Integrated spectrum         #
	#-------------------------------------------------#
	if cfg_par['mcmc_pars'].get('enable', False) == False :
		print '...start binning...\n'

	#select velocities and fluxes which belong to the disk
	VROT = cfg_par['vel_pars'].get('vrot', 200)
	vel_index = ((vel_zoom >= -VROT) & (vel_zoom <= VROT))

	#straighten the array

	lin_vel = vel_zoom[vel_index]
	lin_flux = flux_zoom[vel_index]

	#determine the integrated spectrum
	spec = np.zeros([len(vels)])
	for i in xrange(0, len(vels)-1):
		#look for the right velocity bin
		index = (vels[i] <= lin_vel) & (lin_vel < vels[i+1])

		#update the flux bin
		spec[i] = -np.sum(lin_flux[index])

	if cfg_par['mcmc_pars'].get('enable', False) == False :
		print '...end binning...\n'

	#-------------------------------------------------#
	# CLEAN the CUBE of useless values                #
	#-------------------------------------------------#
	#clean the flux and velocity cube after the interpolation
	#for plotting:
	
	#absorbed part of the disk
	vel_ind = velocity == -np.inf
	velocity[vel_ind] = np.nan
	
	#disk in front of continuum
	disk_ind = disk_front == -999.
	disk_front[disk_ind] = np.nan
	
	#disk behind continuum
	disk_ind = disk_behind == -999.
	disk_behind[disk_ind] = np.nan
	
	return spec, velocity, disk_front, disk_behind