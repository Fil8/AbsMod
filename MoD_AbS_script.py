
# coding: utf-8

# # MoD AbS

# In[5]:

##################################################
# #                 MoD AbS                    # #
# #     modelling disks for HI absorption      # #
# #      fit the line and writes a table       # #
#                                                #
# Filippo Maccagni 03 - 03 - 2017                #
# ASTRON - Kapteyn Institute                     #
##################################################

# import installed python modules
#!/usr/bin/env python
import math
import time
import numpy as np
import numpy.ma as ma
import string
import os
import sys
import pyfits
from astropy import wcs
from astropy.io import fits
from scipy.ndimage.interpolation import zoom
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import matplotlib
matplotlib.use('Agg')

import corner
RUN=1
#load magic python notebook module
#get_ipython().magic(u'pylab inline')

print'Modules Imported'

###################################################


# ** PATHS **
#     - Root directory
#     - Output directory (created if non-existing)
#     - Path to continuum image
#     - Path to spectrum

# In[6]:


#-------------------------------------------------#
#Directories                                      # 
#-------------------------------------------------#

#Galaxy Name
GAL = sys.argv[1]

#Root Directory
#directory where the program is located
rootdir = os.path.abspath('')
# rootdir = '/Users/maccagni/Projects/MoD_AbS/data'

#Directory for the input files
root_obs = rootdir+'/'+GAL+'/'

#Directory for the output files
root_out = root_obs+'output_'+GAL+'/'

try:
    os.stat(root_out)
except:
    os.mkdir(root_out)

#-------------------------------------------------#
#Parameter file                                   #
#-------------------------------------------------#

fileinput = root_obs+'par_'+GAL+'.txt'
 
#-------------------------------------------------#
#Continuum .fits file                             #
#-------------------------------------------------#

filecont = root_obs+GAL+'.fits'
#-------------------------------------------------#
#Observed spectrum ASCII file                     #
#-------------------------------------------------#

#spectrum x-axis in Hz
filespec = root_obs+'spec_'+GAL+'.txt' 

#-------------------------------------------------#
#Output table                                     #
#-------------------------------------------------#

out_table = root_out+'/table_out.csv'

#-------------------------------------------------#
#CONSTANTS                                        #
#-------------------------------------------------#

RAD2DEG = 180./math.pi
HI_hz = 1.42040575177e9
C = 2.99792458E5 

###################################################


# ** Basic Functions **
#     - Read parameter file
#     - Write output log table
#     - Convert HMS -> degrees
#     - Convert DMS -> degrees
#     - Convolve with a Gaussian
#     - Measure FWHM & FW20

# In[7]:


#-------------------------------------------------#
#Read the parameter file and create a dictionary  #
#-------------------------------------------------#


def readfile(parameter_file):
        
    parameters = {}
    
    try:
        param_file = open(parameter_file)
    except:
        print "%s file not found"% parameter_file
        return 1
        
    param_list = param_file.readlines()
    param_file.close()     

    for line in param_list:
        if line.strip():  # non-empty line?
            tmp = line.split('=')
            tmp2 = tmp[0].split('[')
            key, value = tmp2[0], tmp[-1]  # None means 'all whitespace', the default
            parameters[key] = value
    
    return parameters

#-------------------------------------------------#
#Write the output table                           #
#-------------------------------------------------#


def write_table(out_table):

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
    line_3 = str(RMAX)+''','''+str(RMIN)+''','''+str(H_0)+','+str(I)+','+str(PA)+''','''
    line_4 = str(RMAXD2)+''','''+str(RMIND2)+''','''+str(H_0D2)+''','''+str(ID2)+''','''+str(PAD2)+''','''  
    line_5 = str(VROT)+''','''+str(SIGN)+''','''+str(2*DISP)+''',''' 
    line_6 = str(RES)+''','''+str(RES_FIN)+''','''+str(CHI_SQ)+'''\n'''

    line = line_1+line_2+line_3+line_4+line_5+line_6
    table_out.write(line) 
    
    table_out.close()

    return RUN

#-------------------------------------------------#
#Coordinates converter                            #
#-------------------------------------------------#


# HMS -> degrees
def ra2deg(ra_degrees):
    
        ra = string.split(ra_degrees, ':')
    
        hh = float(ra[0])*15
        mm = (float(ra[1])/60)*15
        ss = (float(ra[2])/3600)*15
        
        return hh+mm+ss


# DMS -> degrees
def dec2deg(dec_degrees):
        dec = string.split(dec_degrees, ':')
        
        hh = abs(float(dec[0]))
        mm = float(dec[1])/60
        ss = float(dec[2])/3600
        
        return hh+mm+ss

#-------------------------------------------------#
# Size converter                                  #
#-------------------------------------------------#


# Angle -> radius [Mpc]
def ang2lin(z, dl, ang): 
    # r in arcsec   
    #dl = dl/3.085678e24 # Mpc
    r = ang * dl / (RAD2DEG * 3600 * (1+z)**2) # Mpc
    
    return r


# radius -> angle [arcsec]
def lin2ang(z, dl, r): 
    # r in Mpc

    ang = RAD2DEG * 3600. * r * (1.+z)**2 / dl # arcsec

    return ang

#-------------------------------------------------#
# Gaussian Convolution                            #
#-------------------------------------------------#


def smooth_1d(convo):
        arg = -((vels*vels)/(2*DISP*DISP))
        gauss = 1./(np.sqrt(2*np.pi)*DISP)*np.exp(arg)
        convolved_func = np.convolve(convo, gauss, mode='same')
         
        return convolved_func
 
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
 

###################################################


# ** FUNCTIONS of the MODEL **
# 
#     - Load continuum image and interpolate to initial resolution
#     - Define the disk
#     - Compute integrated absorption line for each disk and merge them together

# In[12]:


#-------------------------------------------------#
#Continuum CUBE: Input for the absorption        #
#-------------------------------------------------#


def build_continuum(x_los, y_los):
        
        #Load continuum: 
        f = pyfits.open(filecont)
        dati = f[0].data
        head = f[0].header
        dati = np.squeeze(dati)
        dati = np.squeeze(dati)
        # define the resolution of the continuum image
        scale_cont_asec = head['CDELT2']*3600
        scale_cont_pc = ang2lin(z_red, D_L, scale_cont_asec)*1e6

        #load the continuum image
        head = fits.getheader(filecont)

        del head['CTYPE4']
        del head['CDELT4']    
        del head['CRVAL4']
        del head['CRPIX4']
        del head['CRPIX3'] 
        del head['CRVAL3']
        del head['CDELT3']
        del head['CTYPE3']
        del head['NAXIS3']
        del head['NAXIS4']        
        del head['NAXIS']
        del head['CROTA1']
        del head['CROTA2']
        del head['CROTA3']
        del head['CROTA4']

        w = wcs.WCS(head)    
        #convert coordinates in pixels
        #cen_x,cen_y=w.wcs_world2pix(ra,dec,0)
        cen_x, cen_y = w.wcs_world2pix(ra, dec, 1)
        
        print '\tContinuum centre [pixel]:\t'+'x: '+str(cen_x)+'\ty: '+str(cen_y) 
        print '\tContinuum pixel size [pc]:\t'+str(scale_cont_pc)+'\n'
              
        #deterimne the edges of the output cube 
        #on the continuum image
        x_los_num_right = x_los[-1]/scale_cont_pc
        x_los_num_left = x_los[0]/scale_cont_pc
        y_los_num_up = y_los[-1]/scale_cont_pc
        y_los_num_low = y_los[0]/scale_cont_pc
        y_up = cen_y+y_los_num_up
        x_right = cen_x+x_los_num_right
        y_low = cen_y+y_los_num_low
        x_left = cen_x+x_los_num_left
        #approximate
        x_left_int = math.modf(x_left)
        x_right_int = math.modf(x_right)
        y_low_int = math.modf(y_low)
        y_up_int = math.modf(y_up)
        #select the continuum subset
        yshape = int(y_up_int[1] - y_low_int[1])
        xshape = int(x_right_int[1] - x_left_int[1])
        
        sub_dati = np.zeros([yshape, xshape])

        #Top Left corner
        if cen_y > head['NAXIS2']/2 and cen_x < head['NAXIS1']/2:

            diff_x = int(cen_x*2)
            diff_y = int(head['NAXIS2']-cen_y)

            subcont = dati[diff_y:head['NAXIS2'],0:diff_x]

        #Top Right corner
        if cen_y > head['NAXIS2']/2 and cen_x > head['NAXIS1']/2:

            diff_x = int(head['NAXIS1']-cen_x)
            diff_y = int(head['NAXIS2']-cen_y)

            subcont = dati[diff_y:head['NAXIS2'],diff_x:head['NAXIS2']]

        #Bottom Left corner
        if cen_y < head['NAXIS2']/2 and cen_x < head['NAXIS1']/2:
            diff_x = int(cen_x*2)
            diff_y = int(cen_y*2)

            subcont = dati[0:diff_y,0:diff_x]


        #Bottom right corner
        if cen_y < head['NAXIS2']/2 and cen_x > head['NAXIS1']/2:

            diff_x = int(head['NAXIS1']-cen_x)
            diff_y = int(cen_y*2)

            subcont = dati[0:diff_y,diff_x:head['NAXIS2']]

        rows = head['NAXIS2']- subcont.shape[0]

        zerows = np.zeros([rows/2, subcont.shape[1]])
        subcont = np.vstack([subcont,zerows])
        subcont = subcont[::-1,:]
        subcont = np.vstack([subcont,zerows])
        subcont = subcont[::-1,:]

        columns = head['NAXIS1']- subcont.shape[1]
        zercolumn = np.zeros([head['NAXIS2'],columns/2])
        subcont = np.hstack([subcont,zercolumn])
        subcont = subcont[:,::-1]
        subcont = np.hstack([subcont,zercolumn])
        subcont = subcont[:,::-1]
  
        #determine how much I have to interpolate     
        zoom_factor = float(len(x_los))/float(len(subcont[0]))
        #interpolate to the desired resolution of the cycle 1 cube
        zoom_dati = zoom(subcont, zoom_factor, order=3)

        return zoom_dati
    
#-------------------------------------------------#
#Function for the DISK coordinates and velocity   #
#-------------------------------------------------#


def space(z_cube, y_cube, x_cube, continuum_cube, rmax, rmin, h_0, pa, i):  
    
        #-------------------------------------------------#
        #takes an input cube in space coordinates and the # 
        #continuum cube returns the cube of velocities    #
        #of the disk and the continuum cube with values   # 
        # only where there is absorption                  #
        #-------------------------------------------------#

        #trigonometric parameters of the disk
        i_rad = math.radians(i)
        pa_rad = math.radians(pa)

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
        vel = -SIGN*sin_i*np.cos(angle)*VROT
        #for plotting: define the disk in front of the continuum
        disk_front = vel.copy()
        index_vel = (continuum_cube == 0.0)
        #if np.all(index_vel) != False:
        vel[index_vel] = -np.inf
        #condition for DISK 1
        idx = ((r > rmax) | (r < rmin) | (abs(z) >= h_0/2.))
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

def mod_abs(z_cube, y_cube, x_cube, continuum_cube_z, flag,vels):
 
    #-------------------------------------------------#
    # Load the parameters for the right disk          #
    #-------------------------------------------------#       
    #flag=2 inner second disk
    if flag == int(2):
        rmax = float(par.get('rmax_in'))
        rmin = float(par.get('rmin_in'))
        h_0 = float(par.get('h0_in'))
        i = float(par.get('i_in'))
        pa = float(par.get('pa_in'))    

    #flag=1 outer first (or only) disk
    if flag == int(1):
        rmax = float(par.get('rmax'))
        rmin = float(par.get('rmin'))
        h_0 = float(par.get('h0'))
        i = float(par.get('i'))
        pa = float(par.get('pa'))
        
    #-------------------------------------------------#
    # VELOCITY and FLUX of the absorbed disk          #
    #-------------------------------------------------#                    
    
    velocity, flusso, disk_front, disk_behind =         space(z_cube, y_cube, x_cube, continuum_cube_z, rmax, rmin, h_0, pa, i)
    
    #-------------------------------------------------#
    # INTERPOLATE to final RESOLUTION                 #
    #-------------------------------------------------#
    
    print '...start interpolation...\n'

    #determine the factor for the interpolation depending on 
    #the specified resolutions
    
    factor = RES/RES_FIN
    print '\tInterpolation of a factor:\t'+str(factor)+'\n'
    #print np.ma.masked_invalid(velocity).sum()
    #increase the order for non-linear interpolation
    vel_zoom = zoom(velocity, factor, order=ORDER)
    flux_zoom = zoom(flusso, factor, order=ORDER)

    print '...end interpolation...\n'

    #-------------------------------------------------#
    # BIN the CUBE in the Integrated spectrum         #
    #-------------------------------------------------#
    
    print '...start binning...\n'
    #select velocities and fluxes which belong to the disk
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


#-------------------------------------------------#
#CHI2 and RESIDUAL spectrum                       #
#-------------------------------------------------#
def chi_res(s_mod, s_obs):
    
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
    

###################################################


# ** Input PARAMETERS of the DISK **
#     - variables from the parameter file
#     - velocity array
#     - cube
#     - output image filename

# In[13]:


###### -------------------------------------------------#
#SET the VARIABLES                                #
#-------------------------------------------------#

#Call the function and set the dictionary
par = readfile(fileinput)

# Disk 1 
RMAX = float(par.get('rmax'))                   #Maximum radius [pc]
RMIN = float(par.get('rmin'))                   #Minimum radius [pc]
H_0 = float(par.get('h0'))                      #Thickness [pc]
I = float(par.get('i'))                         #Inclination [degrees]
PA = float(par.get('pa'))                     #Position angle [degrees]
PA = PA+90.
par['pa'] = PA

#Disk 2 
# !!! RMAXD2 < RMAX !!! #
RMAXD2 = float(par.get('rmax_in'))
RMIND2 = float(par.get('rmin_in'))
H_0D2 = float(par.get('h0_in'))
ID2 = float(par.get('i_in'))
PAD2 = float(par.get('pa_in'))
PAD2 = PAD2+90.
par['pa_in'] = PAD2
# Rotation Curve
VROT = float(par.get('vrot'))                   #Flat limit of the rotation curve [km/s]
SIGN = float(par.get('sign'))                   #Direction of rotation [=/- 1]

# Continuum information
PA_C = float(par.get('pa_cont'))                #Position angle of the continuum [degrees]
CONT_LIM = float(par.get('flux_cont_lim'))      #Noise limit: sets the region of the absorption !!!
VSYS = float(par.get('v_sys'))                  #Systemic velocity [km/s]
D_L = float(par.get('d_l'))                     #luminosity distance [Mpc]
z_red = float(par.get('z'))                     #redshift
RA = par.get('ra')                              #Right Ascention 
DEC = par.get('dec')                            #Declination

RA = string.strip(RA)
DEC = string.strip(DEC)

# Resolution Information
RES = float(par.get('pix_res'))                 #Resolution 1st cycle  [pc]
RES_FIN = float(par.get('pix_res_fin'))         #Final resoluion   [pc]
VRES = float(par.get('vel_res'))                #Velocity resolution for binning [km/s]
DISP = float(par.get('disp'))                   #Dispertion for final resolution (~ to observed spectrum) [km/s]
ORDER = int(par.get('order'))                   #Order of spline for interpolation

#MCMC parameters
DIM_mcmc = int(par.get('ndim_mcmc'))
WALK_mcmc = int(par.get('nwalkers_mcmc'))
STEPS_MCMC = int(par.get('nsteps_mcmc'))

I_d = float(par.get('I_left'))
I_u = float(par.get('I_right'))

PA_d = float(par.get('PA_left'))
PA_u = float(par.get('PA_right'))


#galaxy name in the parameter dictionary
par['gal'] = GAL

#-------------------------------------------------#
#Print variables                                  #
#-------------------------------------------------#

print '********************\n'
print 'INPUT PARAMETERS\n'
print '********************\n'
for keys, values in par.items():
    print keys+' = '+str(values)
print '********************\n'   

#-------------------------------------------------#
#Convert coordinates                              #
#-------------------------------------------------#

#convert to degrees
ra = ra2deg(RA)
dec = dec2deg(DEC)
#convert degrees to radians
PA_C_rad = math.radians(PA_C)

#-------------------------------------------------#
#CUBE                                             #
#-------------------------------------------------#

print 'INPUT DISK\n'
print '********************\n'   

#build the cube which contains my disk based on its (RMAX)
#and on the resolution of the 1st cycle

#set the edges to RMAX+20%
x_los = np.arange(-RMAX*1.2, +RMAX*1.2+RES, RES)
y_los = np.arange(-RMAX*1.2, +RMAX*1.2+RES, RES)
z_los = np.arange(-RMAX*1.2, +RMAX*1.2+RES, RES)

#edges of the cube
print 'edges of the cube [pc]:\t\t\t'+str(x_los[-1]), str(x_los[0])
print 'size of the cube [pixels]]:\t\t'+str(len(x_los))+' x '      +str(len(y_los))+' x '+str(len(z_los))
print 'resolution first cycle [pc]:\t\t'+str(RES)+'\n'

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

#load observed spectrum
spec_obs = np.loadtxt(filespec)
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
print 'edges of the velocity array [km/s]:\t' + str(vels[-1]), str(vels[0])
print 'velocity resolution [km/s]:\t\t' + str(VRES)
print 'convolved velocity resolution [km/s]:\t' + str(2.*DISP)+'\n'
print '********************\n'


# ** MAIN **
#     - create cube of continuum
#     - create cube of coordinates
#     - compute integrated spectrum
#     - compute cubes of
#         - absorbed disk
#         - disk in front of continuum
#         - disk behind continuum
#     - convolve integrated spectrum
#     - normalize spectrum to peak of observed spectrum

# In[14]:


#-------------------------------------------------#
#MAIN MAIN MAIN                                   #
#-------------------------------------------------#

print '********************\n'   
print '...Begin...\n'

#start timer
tempoinit = time.time()

#-------------------------------------------------#
#Set the input continuum cube                     # 
#-------------------------------------------------#

print '... set continuum input cube...\n'

#load the continuum image interpolated at the specified resolution
continuum_image = build_continuum(y_los, x_los)

#to use for uniform absorption 
#continuum_image=np.zeros([len(x_los),len(y_los)])+1.

#create a cube containing the fluxes of the continuum image

#-------------------------------------------------#
# Cube with the continuum image at z=0            # 
# the axis are sorted in the array: [y,z,x]       #
#-------------------------------------------------#

continuum_cube_z = np.dstack([continuum_image]*(len(z_los)))
continuum_cube_z = np.swapaxes(continuum_cube_z, 1, 2)

#mask the noise of the continuum in the cube
index_mask = continuum_cube_z < CONT_LIM
continuum_cube_z[index_mask] = 0.0
continuum_cube_z2 = continuum_cube_z.copy()

#mask the noise of the continuum in the continuum image
index_mask = continuum_image < CONT_LIM        
continuum_image[index_mask] = 0.0

#index_mask = continuum_image != 0.0        
#continuum_image[index_mask] = 1.0

#-------------------------------------------------#
#Set the input coordinates cube                  #
#-------------------------------------------------#

print '...set disk...\n'

#create a meshgrid for the cube
z_cube, y_cube, x_cube = np.meshgrid(z_los, y_los, x_los)

#-------------------------------------------------#
#Compute absorption                              #
#-------------------------------------------------#

print '...compute absorption for disk 1...\n'
#compute the spectrum and the cubes
spec_integral, velocity, disk_front, disk_behind =    mod_abs(z_cube, y_cube, x_cube, continuum_cube_z, int(1),vels)

#-------------------------------------------------#
#2 DISK CASE                                     #
#-------------------------------------------------#

if RMAXD2 != 0.0 :
    print '...compute absorption for disk 2...\n'

    #compute the second set of lines and cubes and 
    #merge them with the first set
    spec_int_2, velocity_2, disk_front_2, disk_behind_2 =        mod_abs(z_cube, y_cube, x_cube, continuum_cube_z2, int(2),vels)
    
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
#CONVOLVE the SPECTRUM                            #
#-------------------------------------------------#

#print '...convolve spectrum...\n'

#convolve the spectrum at the desired velocity resolution
#spec_int[1, :] = smooth_1d(spec_int[1, :])

#-------------------------------------------------#
#NORMALIZE the SPECTRUM to the OBSERVED one       #
#-------------------------------------------------#
    
print '...normalize spectrum...\n'
        
#normalize modelled spectrum to observed one
peak_obs = np.min(spec_obs[:, 1])
peak_mod = np.min(spec_int[1, :])

spec_int_norm = np.divide(spec_int[1, :], peak_mod)
spec_int_mod = np.multiply(spec_int_norm, peak_obs)
#spec_int_mod = spec_int[1,:].copy()
#-------------------------------------------------#
#Determine residuals & chi2 & widths of the lines #
#-------------------------------------------------#

print '...compute some stats...\n'

spec_full = np.array([vels, spec_int_mod])
#spec_full[1,:]=0.0


residuals, obs_res, mod_res, CHI_SQ = chi_res(spec_full, spec_obs)

FWHM, FW20 = widths(spec_full)

#-------------------------------------------------#
#Write new line in output table                   #
#-------------------------------------------------#
print '...write table some stats...\n'

RUN = write_table(out_table)

print '...End... \n'

tempofin = (time.time()-tempoinit)/60.
tempoinit = tempofin

print '********************'   
print "\tTotal time: %f minutes" % tempofin  

print '********************'   
print 'NORMAL TERMINATION'
print '********************\n' 

###################################################


# In[15]:


#-------------------------------------------------#
# PLOT function                                   #
#-------------------------------------------------#


def plot_figure(spec_obs, spec_plt, res, outfile_fig,
                flusso, continuum_image, disk_front, disk_behind):

    plt.ioff()

    #define figure parameters  
    params = {'legend.fontsize': 18,
              'axes.linewidth': 3,
              'axes.labelsize': 22,
              'lines.linewidth': 1,
              'xtick.labelsize': 22,
              'ytick.labelsize': 22,
              'xtick.major.size': 10,
              'xtick.major.width': 4,
              'xtick.minor.size': 1,
              'xtick.minor.width': 1,
              'ytick.major.size': 10,
              'ytick.major.width': 4,
              'ytick.minor.size': 4,
              'ytick.minor.width': 1,
              'text.usetex': True,
              'text.latex.unicode': True}
    rc('font', **{'family': 'serif', 'serif': ['serif']})
    plt.rcParams.update(params)

    #-------------------------------------------------#
    # Set arrays of ticks                             #
    #-------------------------------------------------#  
    #spectrum
    #limits [km/s]
    xleft = -1265
    xright = 1265

    y_massimo = np.max([np.max(spec_obs[:, 1]), np.max(spec_plt[1, :]), np.max(res[1, :])])
    y_minimo = np.min([np.min(spec_obs[:, 1]), np.min(spec_plt[1, :]), np.min(res[1, :])])

    yup_spec = y_massimo + y_massimo * 0.1
    ydown_spec = y_minimo + y_minimo * 0.1

    # to plot residuals separately
    yup_res = np.max(res[1, :])+np.max(res[1, :])*0.1
    ydown_res = np.min(res[1, :])+np.min(res[1, :])*0.2
    x_tick_array = [-1250, -1000, -750, -500, -250, 0, 250, 500, 750, 1000, 1250]
    x_tick_labels_array = [str(-1250),str(-1000),str(-750), str(-500), str(-250), str(0), str(250), str(500), str(750),str(1000),str(1250)]
    x_size_label = '[mJy]'
    #cube
    ticks_array = [-RMAX, -RMAX / 2., 0, RMAX / 2, RMAX]
    print ticks_array
    if RMAX > 1000.:
        tick_labels_array = [str(-RMAX / 1e3), str(-RMAX / 2e3), str(0.0),
                             str(RMAX / 2e3), str(RMAX / 1e3)]
        size_label = '[kpc]'
    if RMAX <= 1000.:
        tick_labels_array = [str(-RMAX), str(-RMAX / 2.), str(0.0),
                             str(RMAX / 2.), str(int(RMAX))]
        size_label = '[pc]'

        #-------------------------------------------------#
    # Set FIGURE and GRID                             #
    #-------------------------------------------------#

    fig_a = plt.figure(figsize=(18, 18), dpi=100)

    #set the FULL grid
    gs_all = gridspec.GridSpec(2, 1)
    gs_all.update(left=0.1, right=0.9, wspace=0.0, hspace=0.00)

    #set the grid for the spectrum and the projected cube
    gs_spec = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_all[0],
                                               wspace=0.0, hspace=0.0)
    
    
    
    gs_ort = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_all[1],
                                              wspace=0.2, hspace=0.0)

    #-------------------------------------------------#
    # Plot the spectrum and parameters of the disk    #
    #-------------------------------------------------#

    #define plots
    ax_spec = fig_a.add_subplot(gs_spec[0:2, 0:3])
    ax_res = fig_a.add_subplot(gs_spec[2, 0:3])

    #-------------------------------------------------#
    # Spectrum                                        #
    #-------------------------------------------------#  

    #plot the observed and modelled spectra
    ax_spec.plot(spec_obs[:, 0], spec_obs[:, 1], ls='-', c='black',
                 label=r'observation', marker=' ', lw=3)
    ax_spec.plot(spec_plt[0, :], spec_plt[1, :], ls='-', c='red', label=r'model', marker=' ', lw=3)

    #set limits of image
    ax_spec.set_xlim(xleft, xright)
    ax_spec.set_ylim(ydown_spec, yup_spec)

    #plot horizontal line at zero
    xx = [xleft, xright]
    yy = [0., 0.]
    ax_spec.plot(xx, yy, ls='--', lw=1, color='black')

    #plot vertical line at zero
    xx = [0., 0.]
    yy = [ydown_spec, yup_spec]
    ax_spec.plot(xx, yy, ls='--', lw=1, color='black')

    #set ticks
    #ax_spec.set_xticks(x_tick_array)
    #ax_spec.set_xticklabels(x_tick_labels_array)

    #set labels of spectrum
    ax_spec.set_ylabel(r'Flux\, ' + x_size_label)

    #set legend
    #ax_spec.legend(loc=4)

    #-------------------------------------------------#
    # Residuals                                       #
    #-------------------------------------------------#  

    #enlarge residuals    
    idx_right = np.where(np.abs(spec_plt[0,:] - VROT) == np.abs(spec_plt[0,:] - VROT).min())[0]
    idx_left = np.where(np.abs(spec_plt[0,:] - (-VROT)) == np.abs(spec_plt[0,:] - -(VROT)).min())[0]

    #res_left= spec_plt[:,0:idx_left].copy()
    #res_right= spec_plt[:,idx_right:-1].copy()
   
    #res_left[1,:] = spec_obs[0:idx_left,1]
    #res_right[1,:] = spec_obs[idx_right-1:-1,1]
    
    res_tot = res # np.hstack([res_left,res,res_right])
    #plot residuals and observed spectrum
    #ax_res.plot(res_enl[:, 0], res_enl[:, 1], ls='-', c='orange', 
    #            label=r'resitudals', marker=' ', lw=3)
    #ax_spec.plot(res_enl[:, 0], res_enl[:, 1], ls='-', c='orange',
    #             label=r'residuals', marker=' ', lw=3)
    #ax_spec.plot(spec_plt[0, :], spec_plt[1, :], ls='-', c='red', label=None, marker=' ', lw=3)
    ax_res.plot(res_tot[0, :], res_tot[ 1, :], ls='-', c='orange', 
                label=r'residuals', marker=' ', lw=3)
    
    
    # to plot residuals separately
    #set limits of image
    ax_res.set_xlim(xleft, xright)
    ax_res.set_ylim(ydown_res, yup_res)

    #plot horizontal & vertical line at zero line at zero
    xx = [xleft, xright]
    yy = [0., 0.]
    ax_res.plot(xx, yy, ls='--', lw=1, color='black')
    xx = [0., 0.]
    yy = [ydown_res, yup_res]
    ax_res.plot(xx, yy, ls='--', lw=1, color='black')

    #plot vertical line at edges of rotation curve
    #xx = [res[0, 0], res[0, 0]]
    #yy = [ydown, yup]
    #ax_res.plot(xx, yy, ls='--', lw=1, color='black')   
    #xx = [res[0, 0-1], res[0, -1]]
    #yy = [ydown, yup]
    #ax_res.plot(xx, yy, ls='--', lw=1, color='black') 

    #set labels of spectrum
    ax_spec.set_xlabel(r'Velocity [km\,s$^{-1}$]')

    #set ticks
    #to plot residuals separately
    ax_res.set_ylabel(r'Flux\, '+x_size_label)
    ax_res.set_xticks(x_tick_array)
    #ax_res.set_yticks(y_tick_array)  
    ax_res.set_xticklabels(x_tick_labels_array)
    #ax_res.set_yticklabels(tick_labels_array)  
    #set legend
    ax_spec.legend(loc=4)
    ax_res.legend(loc=4)

    #set title
    ax_spec.set_title('Spectrum', fontsize=24)

    #-------------------------------------------------#
    # Plot the cube in different projections          #
    #-------------------------------------------------#

    #define plots
    ax_pv = fig_a.add_subplot(gs_ort[0, 0])
    ax_po = fig_a.add_subplot(gs_ort[0, 1])
    ax_pl = fig_a.add_subplot(gs_ort[0, 2])

    #-------------------------------------------------#
    # PLANE of the SKY                                #
    #-------------------------------------------------#

    #project the fluxes of the absorbed cube
    flux_zoom_pv = np.nanmean(flusso, axis=1)
    disk_front_pv = np.nanmean(disk_front, axis=1)
    disk_behind_pv = np.nanmean(disk_behind, axis=1)

    #plot continuum image
    ax_pv.imshow(continuum_image, extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]],
                 origin='lower', cmap='hot_r', alpha=0.8)

    cont = [CONT_LIM]
    ax_pv.contour(continuum_image, cont, origin='lower', colors='black', linewidths=3,
                  ls='-.', extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]])

    #plot absorbed part of the disk
    ax_pv.imshow(flux_zoom_pv, origin='lower', cmap='nipy_spectral',
                 extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]], alpha=1.,
                 vmin=np.min(spec_obs[:,0]),vmax=np.max(spec_obs[:,0]))
    print np.nanmin(flux_zoom_pv)
    #plot disk in front of continuum
    ax_pv.imshow(disk_front_pv, origin='lower', cmap='nipy_spectral',
                 extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]], alpha=0.1)

    #plot disk in behind continuum
    ax_pv.imshow(disk_behind_pv, origin='lower', cmap='nipy_spectral',
                 extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]], alpha=0.4)

    #set ticks & labels
    ax_pv.set_xlabel(r'x ' + size_label)
    ax_pv.set_ylabel(r'y ' + size_label)
    ax_pv.set(adjustable='box-forced', aspect='equal')

    ax_pv.set_xticks(ticks_array)
    ax_pv.set_yticks(ticks_array)
    ax_pv.set_xticklabels(tick_labels_array)
    ax_pv.set_yticklabels(tick_labels_array)
    #set title
    ax_pv.set_title('Plane of the sky', fontsize=24)

    #-------------------------------------------------#
    # View from above                                 #
    #-------------------------------------------------#

    #project the fluxes of the absorbed cube
    flux_zoom_po = np.nanmean(flusso, axis=0)
    disk_front_po = np.nanmean(disk_front, axis=0)
    disk_behind_po = np.nanmean(disk_behind, axis=0)

    #plot absorbed part of the disk
    ax_po.imshow(flux_zoom_po, origin='lower', cmap='nipy_spectral',
                 extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]], alpha=1.)

    #plot disk in front of continuum
    ax_po.imshow(disk_front_po, origin='lower', cmap='nipy_spectral',
                 extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]], alpha=0.1)
    #plot disk in behind continuum
    ax_po.imshow(disk_behind_po, origin='lower', cmap='nipy_spectral',
                 extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]], alpha=0.4)

    #set ticks
    ax_po.set_xlabel(r'x ' + size_label)
    ax_po.set_ylabel(r'z ' + size_label)
    ax_po.set(adjustable='box-forced', aspect='equal')

    ax_po.set_xticks(ticks_array)
    ax_po.set_yticks(ticks_array)
    ax_po.set_xticklabels(tick_labels_array)
    ax_po.set_yticklabels(tick_labels_array)
    #set title
    ax_po.set_title('''View from `above' ''', fontsize=24)
    ax_po.yaxis.labelpad = -10

    #-------------------------------------------------#
    # View from the side                              #
    #-------------------------------------------------#

    #project the fluxes of the absorbed cube
    flux_zoom_pl = np.nanmean(flusso, axis=2)
    disk_front_pl = np.nanmean(disk_front, axis=2)
    disk_behind_pl = np.nanmean(disk_behind, axis=2)

    #plot absorbed part of the disk
    ax_pl.imshow(flux_zoom_pl, origin='lower', cmap='nipy_spectral',
                 extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]], alpha=1.)

    #plot disk in front of continuum
    ax_pl.imshow(disk_front_pl, origin='lower', cmap='nipy_spectral',
                 extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]], alpha=0.1)
    #plot disk in behind continuum
    ax_pl.imshow(disk_behind_pl, origin='lower', cmap='nipy_spectral',
                 extent=[x_los[0], x_los[-1], y_los[0], y_los[-1]], alpha=0.4)

    #set ticks
    ax_pl.set_xlabel(r'z ' + size_label)
    ax_pl.set_ylabel(r'y ' + size_label)
    ax_pl.set(adjustable='box-forced', aspect='equal')

    ax_pl.set_xticks(ticks_array)
    ax_pl.set_yticks(ticks_array)
    ax_pl.set_xticklabels(tick_labels_array)
    ax_pl.set_yticklabels(tick_labels_array)
    #set title
    ax_pl.set_title('''View from the `side' ''', fontsize=24)
    ax_pl.yaxis.labelpad = -10
    #-------------------------------------------------#
    # Save figure                                     #
    #-------------------------------------------------#   

    fig_a.savefig(outfile_fig, format='png', bbox_inches='tight')

    return 0


# In[16]:


#-------------------------------------------------#
# PLOT                                            #
#-------------------------------------------------#

#-------------------------------------------------#
#Output FIGURE                                    #
#-------------------------------------------------#
#define smart names for the output figure
outfile_fig = root_out+GAL+'_'+str(RUN)+'_'+str(int(I))+'_'+str(int(PA))+'.png'

print '...Begin Plotting...\n'
plot_figure(spec_obs, spec_full, residuals, outfile_fig,
            velocity, continuum_image, disk_front, disk_behind)

print '...End Plotting...\n'

###################################################
# END                                             #
###################################################


# In[69]:


def walkers_plot(walkers, out_fig_walkers_name):
    
    #define figure parameters
    params = {'legend.fontsize': 18,
              'axes.linewidth': 3,
              'axes.labelsize': 22,
              'lines.linewidth': 1,
              'xtick.labelsize': 22,
              'ytick.labelsize': 22,
              'xtick.major.size': 8,
              'xtick.major.width': 2,
              'xtick.minor.size': 1,
              'xtick.minor.width': 1,
              'ytick.major.size': 8,
              'ytick.major.width': 2,
              'ytick.minor.size': 4,
              'ytick.minor.width': 1,
              'text.usetex': True,
              'text.latex.unicode': True}
    rc('font', **{'family': 'serif', 'serif': ['serif']})
    plt.rcParams.update(params)
          
    #-------------------------------------------------#
    # Set FIGURE and GRID                             #
    #-------------------------------------------------#
    
    fig_a = plt.figure(figsize=(12, 8), dpi=100)

    #set the FULL grid
    gs_all = gridspec.GridSpec(2, 1)
    gs_all.update(left=0.1, right=0.9, wspace=0.0, hspace=0.00)
    
    #define subplots
    wal_i = fig_a.add_subplot(gs_all[0, 0])
    wal_pa = fig_a.add_subplot(gs_all[1, 0])
    
    
    #define ticks & labels

    x_tick_array = np.linspace(0, STEPS_MCMC-1, 8, dtype=int)
    
    x_tick_labels_array = x_tick_array.astype(str)
    x_label_name = r'Steps'

    i_tick_array = [0, 15, 30, 45, 60, 75, 90]
    i_tick_labels_array = [str(0), str(15), str(30), str(45), str(60), str(75), str(90)]
    
    #i_tick_array = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    #i_tick_labels_array = [str(0), str(20), str(40), str(60), str(80), str(100), str(120), 
    #                       str(140), str(160), str(180)]
    
    i_label_name = r'I [$^\circ$]'
    
    pa_tick_array = [0, 30, 60, 90, 120, 150, 180]
    pa_tick_labels_array = [str(0), str(30), str(60), str(90), str(120), str(150), str(180)]
    
    #pa_tick_array = [0, 60, 120, 180, 240, 300, 360]
    #pa_tick_labels_array = [str(0), str(60), str(120), str(180), str(240), str(300), str(360)]
    pa_label_name = r'PA [$^\circ$]'
    
    #-------------------------------------------------#
    # walking on I                                    #
    #-------------------------------------------------#
    
    steps_mcmc_array = np.linspace(0, STEPS_MCMC-1, STEPS_MCMC)
    
    for i in xrange(0, walkers.shape[0]):
        
        wal_i.plot(steps_mcmc_array, walkers[i, :, 0], "k", alpha=0.3)
 
    #set labels & ticks
    wal_i.set_xticks(x_tick_array)
    wal_i.set_xticklabels([])    
    
    wal_i.set_yticks(i_tick_array)
    wal_i.set_yticklabels(i_tick_labels_array)    
    wal_i.set_ylabel(i_label_name) 
  
    #-------------------------------------------------#
    # walking on PA                                   #
    #-------------------------------------------------#
    
    for i in xrange(0, walkers.shape[0]):
        
        wal_pa.plot(steps_mcmc_array, walkers[i, :, 1], "k", alpha=0.3)  

    #set labels & ticks
    wal_pa.set_xticks(x_tick_array)
    wal_pa.set_xticklabels(x_tick_labels_array)    
    wal_pa.set_xlabel(x_label_name)
    
    wal_pa.set_yticks(pa_tick_array)
    wal_pa.set_yticklabels(pa_tick_labels_array)    
    wal_pa.set_ylabel(pa_label_name)

    #-------------------------------------------------#
    # Save figure                                     #
    #-------------------------------------------------#   
    
    fig_a.savefig(out_fig_walkers_name, format='pdf', bbox_inches='tight')  

    return 0


def corner_plot(samples, out_fig_samples_name):

    x = samples[:, 0]
    y= samples[:, 1]-90.

    #define levels for contours
    levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # We'll make the 2D histogram to directly estimate the density.

    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=20)
    
    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
    ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
    ])        
              
    #define figure parameters
    params = {'legend.fontsize': 18,
              'axes.linewidth': 3,
              'axes.labelsize': 22,
              'lines.linewidth': 1,
              'xtick.labelsize': 22,
              'ytick.labelsize': 22,
              'xtick.major.size': 8,
              'xtick.major.width': 2,
              'xtick.minor.size': 1,
              'xtick.minor.width': 1,
              'ytick.major.size': 8,
              'ytick.major.width': 2,
              'ytick.minor.size': 4,
              'ytick.minor.width': 1,
              'text.usetex': True,
              'text.latex.unicode': True}
    rc('font', **{'family': 'serif', 'serif': ['serif']})
    plt.rcParams.update(params)
          
    #-------------------------------------------------#
    # Set FIGURE and GRID                             #
    #-------------------------------------------------#

    fig_a = plt.figure(figsize=(12, 12), dpi=100)

    #set the FULL grid
    gs_all = gridspec.GridSpec(2, 2)
    gs_all.update(left=0.05, right=0.95, wspace=0.02, hspace=0.02)

    #define subplots
    ax_i = fig_a.add_subplot(gs_all[0, 0])
    ax_cont = fig_a.add_subplot(gs_all[1, 0])
    ax_pa = fig_a.add_subplot(gs_all[1, 1])
    ax_blank = fig_a.add_subplot(gs_all[0, 1])
    ax_blank.axis('off')

 
    #define colormap
    color = "k"
    red_color = "blue"

    white_cm = LinearSegmentedColormap.from_list("white_cmap", [(1, 1, 1), (1, 1, 1)], N=2) 
    rgba_color = colorConverter.to_rgba(red_color)
    
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)    

    #define ticks & labels
    #x_tick_array = [0, 15, 30, 45, 60, 75, 90]
    #x_tick_labels_array = [str(0), str(15), str(30), str(45), str(60), str(75), str(90)]
    
    x_tick_array = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    x_tick_labels_array = [str(0), str(20), str(40), str(60), str(80), str(100), str(120), 
                           str(140), str(160), str(180)]
    x_label_name = r'I [$^\circ$]'
    
    #y_tick_array = [0, 30, 60, 90, 120, 150, 180]
    #y_tick_labels_array = [str(0), str(30), str(60), str(90), str(120), str(150), str(180)]
    
    y_tick_array = [0, 60, 120, 180, 240, 300, 360]
    y_tick_labels_array = [str(0), str(60), str(120), str(180), str(240), str(300), str(360)]
    y_label_name = r'PA [$^\circ$]'

    #-------------------------------------------------#
    # Contour plot                                    #
    #-------------------------------------------------#

    ax_cont.set_xlim(I_d, I_u)
    ax_cont.set_ylim(PA_d, PA_u)

    ax_cont.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cm, antialiased=False)
    ax_cont.scatter(x, y,  color=color, marker='x', s=10,alpha=0.2)
    ax_cont.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]), 
                     colors=contour_cmap)
    ax_cont.contour(X2, Y2, H2.T, V, colors=color, linewidths=3)

    #set labels & ticks
    ax_cont.set_xticks(x_tick_array)
    ax_cont.set_xticklabels(x_tick_labels_array)    
    ax_cont.set_xlabel(x_label_name)
    
    ax_cont.set_yticks(y_tick_array)
    ax_cont.set_yticklabels(y_tick_labels_array)    
    ax_cont.set_ylabel(y_label_name)
    
    #-------------------------------------------------#
    # Histogram I plot                                #
    #-------------------------------------------------#
    
    #histogram
    I_step = (I_u - I_d) / 90
    bins = np.arange(I_d, I_u+I_step, I_step)

    (n_i, bins_i, patches_i) = ax_i.hist(x, bins, histtype='bar', 
                                         lw=2, 
                                         facecolor='navy', edgecolor='navy')

    #frame limits
    ax_i.set_xlim(I_d, I_u)
    ax_i.set_ylim(0, np.max(n_i)+np.max(n_i)*0.05)
    
    #set labels & ticks
    ax_i.set_xticks(x_tick_array)
    ax_i.set_xticklabels([])     
    
    i_hist_ticks = np.linspace(0, np.max(n_i), 7)
    i_hist_ticks = i_hist_ticks[1::]
    ax_i.set_yticks(i_hist_ticks)
    ax_i.set_ylabel(r'Count')
 
    #-------------------------------------------------#
    # Histogram PA plot                                #
    #-------------------------------------------------#
    
    #histogram
    PA_step = (PA_u - PA_d) / 72
    bins = np.arange(PA_d, PA_u+PA_step, PA_step)

    (n_pa, bins_pa, patches_pa) = ax_pa.hist(y, bins, histtype='bar', 
                                             lw=2, 
                                             facecolor='navy', edgecolor='navy')
    #frame limits
    ax_pa.set_xlim(PA_d, PA_u)
    ax_pa.set_ylim(0, np.max(n_pa)+np.max(n_pa)*0.05)
    
    #set labels & ticks
    ax_pa.set_xticks(y_tick_array)
    ax_pa.set_xticklabels(y_tick_labels_array)    
    ax_pa.set_xlabel(y_label_name)

    pa_hist_ticks = np.linspace(0, np.max(n_pa)+20, 8)
    pa_hist_ticks = pa_hist_ticks[1::]
    ax_pa.set_yticks(pa_hist_ticks)
    #ax_pa.set_yticklabels([])   
    ax_pa.yaxis.tick_right()
    ax_pa.yaxis.set_ticks_position('both')
    
    ax_pa.set_ylabel(r'Count')
    ax_pa.yaxis.set_label_position('right')

    #-------------------------------------------------#
    # Save figure                                     #
    #-------------------------------------------------#   
    
    fig_a.savefig(out_fig_samples_name, format='pdf', bbox_inches='tight')  

    return 0

'''

# In[70]:


#--------------------------------------------------#
# PLOT  MCMC                                       #
#--------------------------------------------------#

#load data
samples_text = root_out+GAL+'_samples.txt'
walkers_array = root_out+GAL+'_walkers.npy'
samples = np.loadtxt(samples_text, dtype=float)
walkers = np.load(walkers_array)


#define smart names for the output figure
out_samples_figure = root_out+GAL+'_'+'_samples1.pdf'
out_walkers_figure = root_out+GAL+'_'+'_walkers1.pdf'
 
#-------------------------------------------------#
#Plot                                             #
#-------------------------------------------------#
print '...Begin Plotting...\n'

walkers_plot(walkers, out_walkers_figure)
#corner_plot(samples, out_samples_figure)

print '...End Plotting...\n'

###################################################


# In[25]:


import corner

samples[:, 1]-=90.
figure = corner.corner(samples)

'''
