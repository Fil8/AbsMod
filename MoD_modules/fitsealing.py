import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy.ndimage.interpolation import zoom

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
	print dati.shape
	# define the resolution of the continuum image
	scale_cont_asec = head['CDELT2']*3600
	scale_cont_pc = conv.ang2lin(z_red, D_L, scale_cont_asec)*1e6

	#load the continuum image
	head = fits.getheader(workdir+filecont)
	head = clean_header(head)
	#print head
	w = wcs.WCS(head)    
	#convert coordinates in pixels
	cen_x,cen_y=w.wcs_world2pix(ra,dec,0)
	#cen_x, cen_y = w.wcs_world2pix(ra, dec, 1)
	cen_x = np.round(cen_x,0)
	cen_y = np.round(cen_y,0)

	print '\tContinuum centre [pixel]:\t'+'x: '+str(cen_x)+'\ty: '+str(cen_y) 
	print '\tContinuum pixel size [pc]:\t'+str(scale_cont_pc)+'\n'
		  
	#deterimne the edges of the output cube 

	#-------------------------------------------------#
	#CUBE of HI disk                                  #
	#-------------------------------------------------#

	x_los, y_los, z_los = disk.main_box(cfg_par)
	print x_los,y_los,z_los
	# #on the continuum image
	# x_los_num_right = x_los[-1]/scale_cont_pc
	# x_los_num_left = x_los[0]/scale_cont_pc
	# y_los_num_up = y_los[-1]/scale_cont_pc
	# y_los_num_low = y_los[0]/scale_cont_pc

	# y_up = cen_y+y_los_num_up
	# x_right = cen_x+x_los_num_right
	# y_low = cen_y+y_los_num_low
	# x_left = cen_x+x_los_num_left
	# #approximate
	# x_left_int = np.modf(x_left)
	# x_right_int = np.modf(x_right)
	# y_low_int = np.modf(y_low)
	# y_up_int = np.modf(y_up)
	# #select the continuum subset
	# yshape = int(y_up_int[1] - y_low_int[1])
	# xshape = int(x_right_int[1] - x_left_int[1])

   #deterimne the edges of the output cube 
	#on the continuum image
	x_los_num_right=x_los[-1]/scale_cont_pc
	x_los_num_left=x_los[0]/scale_cont_pc
	y_los_num_up=y_los[-1]/scale_cont_pc
	y_los_num_low=y_los[0]/scale_cont_pc
	
	y_up=cen_y+y_los_num_up
	x_right=cen_x+x_los_num_right
	y_low=cen_y+y_los_num_low
	x_left=cen_x+x_los_num_left
	
	#approximate
	x_left_int=np.modf(x_left)
	x_right_int=np.modf(x_right)
	y_low_int=np.modf(y_low)
	y_up_int=np.modf(y_up)
	
	#select the continuum subset
	sub_dati=dati[int(y_low_int[1]):int(y_up_int[1]),int(x_left_int[1]):int(x_right_int[1])]

	if sub_dati.shape[0] != sub_dati.shape[1]:
		sub_dati=dati[int(y_low_int[1]):int(y_up_int[1])-1,int(x_left_int[1]):int(x_right_int[1])]
		
	
	#determine how much I have to interpolate     
	zoom_factor= float(len(x_los))/float(len(sub_dati[0]))
	
	#interpolate to the desired resolution of the cycle 1 cube
	zoom_dati=zoom(sub_dati,zoom_factor,order=3)


	#sub_dati = np.zeros([yshape, xshape])

	# #Top Left corner
	# if cen_y > head['NAXIS2']/2 and cen_x < head['NAXIS1']/2:
	# 	print 'topleft'
	# 	diff_x = int(cen_x*2)
	# 	diff_y = int(head['NAXIS2']-cen_y)

	# 	subcont = dati[diff_y:head['NAXIS2'],0:diff_x]

	# #Top Right corner
	# if cen_y > head['NAXIS2']/2 and cen_x > head['NAXIS1']/2:
	# 	print 'topright'

	# 	diff_x = int(head['NAXIS1']-cen_x)
	# 	diff_y = int(head['NAXIS2']-cen_y)

	# 	subcont = dati[diff_y:head['NAXIS2'],diff_x:head['NAXIS2']]

	# #Bottom Left corner
	# if cen_y < head['NAXIS2']/2 and cen_x < head['NAXIS1']/2:
	# 	print 'bottomleft'
		

	# 	diff_x = int(cen_x*2)
	# 	diff_y = int(cen_y*2)

	# 	subcont = dati[0:diff_y,0:diff_x]


	# #Bottom right corner
	# if cen_y < head['NAXIS2']/2 and cen_x > head['NAXIS1']/2:
	# 	print 'bottomright'
		
	# 	diff_x = int(head['NAXIS1']-cen_x)
	# 	diff_y = int(cen_y*2)

	# 	subcont = dati[0:diff_y,diff_x:head['NAXIS2']]

	# if cen_y == head['NAXIS2']/2 and cen_x == head['NAXIS1']/2:

	# 	subcont = dati[:,:]

	# rows = head['NAXIS2']- subcont.shape[0]
	# zerows = np.zeros([rows/2, subcont.shape[1]])
	# subcont = np.vstack([subcont,zerows])
	# subcont = subcont[::-1,:]
	# subcont = np.vstack([subcont,zerows])
	# subcont = subcont[::-1,:]

	# columns = head['NAXIS1']- subcont.shape[1]
	# zercolumn = np.zeros([head['NAXIS2'],columns/2])
	# #zercolumn = np.zeros([head['NAXIS2']-1,columns/2])

	# subcont = np.hstack([subcont,zercolumn])
	# subcont = subcont[:,::-1]
	# subcont = np.hstack([subcont,zercolumn])
	# subcont = subcont[:,::-1]
	# subcont = dati[:,:]

	# #determine how much I have to interpolate   
	# RES = cfg_par['res_pars'].get('pix_res', 100)
	# factor = RES/scale_cont_pc
	# #zoom_factor = float(len(x_los))/float(len(subcont[0]))
	# zoom_factor = factor
	# print zoom_factor
	# #interpolate to the desired resolution of the cycle 1 cube
	# zoom_dati = zoom(subcont, zoom_factor, order=3)
	# print zoom_dati.shape
	#-------------------------------------------------#
	# Cube with the continuum image at z=0            # 
	# the axis are sorted in the array: [y,z,x]       #
	#-------------------------------------------------#

	continuum_image = zoom_dati.copy()

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