import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy.ndimage.interpolation import zoom
import montage_wrapper as montage
from reproject import reproject_exact
import conversions as conv
import disk as disk
#-------------------------------------------------#
#Continuum CUBE: Input for the absorption        #
#-------------------------------------------------#

def clean_header(header):


	if 'CTYPE4' in header:
	#	del header['NAXIS4']        
		del header['CTYPE4']
		del header['CDELT4']    
		del header['CRVAL4']
		del header['CRPIX4']
	if 'CTYPE3' in header:
		del header['CRPIX3'] 
		del header['CRVAL3']
		del header['CDELT3']
		del header['CTYPE3']
	#	del header['NAXIS3']
	
	header['NAXIS'] = 2 

	if 'CROTA1' in header:
		del header['CROTA1']
	if 'CROTA2' in header:
		del header['CROTA2']
	if 'CROTA3' in header:
		del header['CROTA3']
	if 'CROTA4' in header:
		del header['CROTA4']	


	return header

def build_continuum(cfg_par):

	# read important parameters
	workdir = cfg_par['general'].get('work_directory', None)
	filecont = cfg_par['general'].get('continuum_name', None)	
	D_L = cfg_par['cont_pars'].get('d_l', 60.)
	z_red = cfg_par['cont_pars'].get('redshift', 0.014)

	ra_hms = cfg_par['cont_pars'].get('ra', '00:00:00')
	dec_dms = cfg_par['cont_pars'].get('dec', '00:00:00')

	CONT_LIM = cfg_par['abs_pars'].get('flux_cont_lim', 1e-3)


	#convert to degrees
	ra = conv.ra2deg(ra_hms)
	dec = conv.dec2deg(dec_dms)

	#Load continuum: 
	f = fits.open(workdir+filecont)
	dati = f[0].data
	head = f[0].header
	dati = np.squeeze(dati)
	dati = np.squeeze(dati)
	# define the resolution of the continuum image
	RES = cfg_par['res_pars'].get('pix_res', 100)

	scale_cont_asec = head['CDELT2']*3600
	scale_cont_pc = conv.ang2lin(z_red, D_L, scale_cont_asec)*1e6

	scale_cube_deg = conv.lin2ang(z_red,D_L, RES*1e-6)/3600.
	#load the continuum image
	head = fits.getheader(workdir+filecont)
	head = clean_header(head)

  	if 'NAXIS3' in head:
  		del head['NAXIS3']
  	if 'NAXIS4' in head:
  		del head['NAXIS4']	

  	newfilecont = workdir+'reproj_cont'+str(RES)+'.fits'
	fits.writeto(newfilecont,dati,head,overwrite=True)

	#print head
	w = wcs.WCS(head)    
	#convert coordinates in pixels
	cen_x,cen_y=w.wcs_world2pix(ra,dec,0)
	#cen_x, cen_y = w.wcs_world2pix(ra, dec, 1)

	cen_x = np.round(cen_x,0)
	cen_y = np.round(cen_y,0)


	print '\tContinuum centre [pixel]:\t'+'x: '+str(cen_x)+'\ty: '+str(cen_y) 
	print '\tContinuum pixel size [pc]:\t'+str(scale_cont_pc)+'\n'
	print '\t Size of continuum image:\t'+str(head['NAXIS1'])+'\n'  

 	#deterimne the edges of the output cube 

	#-------------------------------------------------#
	#CUBE of HI disk                                  #
	#-------------------------------------------------#

	x_los, y_los, z_los = disk.main_box(cfg_par)
	# #on the continuum image
	x_los_num_right = x_los[-1]/scale_cont_pc
	x_los_num_left = x_los[0]/scale_cont_pc
	y_los_num_up = y_los[-1]/scale_cont_pc
	y_los_num_low = y_los[0]/scale_cont_pc




	y_up = cen_y+y_los_num_up
	x_right = cen_x+x_los_num_right
	y_low = cen_y+y_los_num_low
	x_left = cen_x+x_los_num_left
	#approximate
	x_left_int = np.modf(x_left)
	x_right_int = np.modf(x_right)
	y_low_int = np.modf(y_low)
	y_up_int = np.modf(y_up)

	#select the continuum subset
	yshape = int(y_up_int[1] - y_low_int[1])
	xshape = int(x_right_int[1] - x_left_int[1])


	#if not all(i>0 for i in [x_left_int[1],x_right_int[1],y_low_int[1],y_up_int[1]]):

	print '\t reproject continuum image\n'

	slave = fits.open(workdir+filecont)[0]
	
	#make header		

  	if 'NAXIS3' in head:
  		del head['NAXIS3']
  	if 'NAXIS4' in head:
  		del head['NAXIS4']

  	new_xshape = int(2*x_los[-1]/scale_cont_pc)

  	head['NAXIS1'] = new_xshape
  	head['NAXIS2'] = new_xshape

  	head['CRPIX1'] = int(new_xshape/2.)
  	head['CRPIX2'] = int(new_xshape/2.)+1

  	head['CRVAL1'] = ra
  	head['CRVAL2'] = dec

  	#head['CDELT1'] = -scale_cube_deg
  	#head['CDELT2'] = scale_cube_deg 


	newdati, footprint = reproject_exact(slave,head)
	newdati = np.nan_to_num(newdati)

	fits.writeto(workdir+'reproj_cont'+str(RES)+'.fits', newdati, head, clobber=True)

	zoom_factor= float(len(x_los))/float(new_xshape)
		
	##interpolate to the desired resolution of the cycle 1 cube
	newdati=zoom(newdati,zoom_factor,order=3)

	#-------------------------------------------------#
	# Cube with the continuum image at z=0            # 
	# the axis are sorted in the array: [y,z,x]       #
	#-------------------------------------------------#

	continuum_image =  newdati.copy()
	continuum_cube_z = np.dstack([continuum_image]*(len(z_los)))
	continuum_cube_z = np.swapaxes(continuum_cube_z, 1, 2)
	#mask the noise of the continuum in the cube
	index_mask = continuum_cube_z < CONT_LIM
	continuum_cube_z[index_mask] = 0.0
	continuum_cube_z2 = continuum_cube_z.copy()

	#mask the noise of the continuum in the continuum image
	index_mask = continuum_image < CONT_LIM        
	continuum_image[index_mask] = 0.0


	return continuum_image, continuum_cube_z