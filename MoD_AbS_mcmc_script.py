##################################################
# #                 MoD AbS                    # #
# #     modelling disks for HI absorption      # #
# #      fit the line in a MCMC simulation     # #
#                                                #
# Filippo Maccagni 07 - 03 - 2017                #
# ASTRON - Kapteyn Institute                     #
##################################################

# import installed python modules
# !/usr/bin/env python
import math
import time
import numpy as np
import random
import string
import sys
import os
import pyfits
import corner
import emcee
from astropy import wcs
from astropy.io import fits
from scipy.ndimage.interpolation import zoom
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

print'Modules Imported'

# -------------------------------------------------#
# Read Inputs                                      #
# -------------------------------------------------#

# Galaxy name
GAL = sys.argv[1]
# Root Directory
rootdir = os.path.abspath('')

# -------------------------------------------------#
# Directories                                      #
# -------------------------------------------------#

# Directory for the input files
root_obs = rootdir + '/' + GAL + '/'
print root_obs
# Directory for the output files
root_out = root_obs + '/output_' + GAL + '/'
try:
    os.stat(root_out)
except:
    os.mkdir(root_out)

# -------------------------------------------------#
# Output files                                     #
# -------------------------------------------------#

out_samples_text = root_out + GAL + '_samples.txt'
out_walkers_vec = root_out + GAL + '_walkers.npy'

out_walkers_figure = root_out + GAL + '_walkers.pdf'
out_samples_figure = root_out + GAL + 'samples.pdf'

out_table = root_out + GAL+'_table_mcmc.csv'

# -------------------------------------------------#
# Parameter file                                   #
# -------------------------------------------------#

fileinput = root_obs + 'par_' + GAL + '.txt'
print fileinput
# -------------------------------------------------#
# Continuum .fits file                             #
# -------------------------------------------------#

filecont = root_obs + GAL + '.fits'

# -------------------------------------------------#
# Observed spectrum ASCII file                     #
# -------------------------------------------------#

filespec = root_obs + 'spec_' + GAL + '.txt'
# input spectrum x-axis in Hz

# -------------------------------------------------#
# CONSTANTS                                        #
# -------------------------------------------------#

RAD2DEG = 180. / math.pi
HI_hz = 1.42040575177e9
C = 2.99792458E5


# -------------------------------------------------#
# BASIC FUNCTIONS                                  #
# -------------------------------------------------#


def readfile(parameter_file):
    parameters = {}

    try:
        param_file = open(parameter_file)
    except:
        print "%s file not found" % parameter_file
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


# -------------------------------------------------#
# Coordinates converter                            #
# -------------------------------------------------#

# HMS -> degrees
def ra2deg(ra_d):
    ra_split = string.split(ra_d, ':')

    hh = float(ra_split[0]) * 15
    mm = (float(ra_split[1]) / 60) * 15
    ss = (float(ra_split[2]) / 3600) * 15

    return hh + mm + ss


# DMS -> degrees
def dec2deg(dec_d):
    dec_split = string.split(dec_d, ':')

    hh = abs(float(dec_split[0]))
    mm = float(dec_split[1]) / 60
    ss = float(dec_split[2]) / 3600
    return hh + mm + ss


# -------------------------------------------------#
# Size converter                                  #
# -------------------------------------------------#

# Angle -> radius [Mpc]
def ang2lin(z, dl, ang):  # r in arcsec

    # dl = dl/3.085678e24 # Mpc
    r = ang * dl / (RAD2DEG * 3600 * (1 + z) ** 2)  # Mpc

    return r


# radius -> angle [arcsec]
def lin2ang(z, dl, r):  # r in Mpc

    ang = RAD2DEG * 3600. * r * (1. + z) ** 2 / dl  # arcsec

    return ang


# -------------------------------------------------#
# Gaussian Convolution                             #
# -------------------------------------------------#

def smoothing_func(conv_spec):
    arg = -((vels * vels) / (2 * DISP * DISP))
    gauss = 1. / (np.sqrt(2 * np.pi) * DISP) * np.exp(arg)
    convolved_pdf = np.convolve(conv_spec, gauss, mode='same')

    return convolved_pdf


# -------------------------------------------------#
# FWHM & FW20                                      #
# -------------------------------------------------#


def widths(spec_model):
    # find peak of the line
    array = spec_model[1, :].copy()
    array_vels = spec_model[0, :].copy()
    minimum = np.min(array)
    minimum_idx = np.where(np.abs(array - minimum) == np.abs(array - minimum).min())[0]

    # FWHM
    left_array = array[0: minimum_idx]
    right_array = array[minimum_idx: len(array)]

    fwhm_min = minimum / 2.

    left_fwhm_idx = np.max(np.where(np.abs(left_array - fwhm_min) ==
                                    np.abs(left_array - fwhm_min).min())[0])
    right_fwhm_idx = np.min(np.where(np.abs(right_array - fwhm_min) ==
                                     np.abs(right_array - fwhm_min).min())[0])

    vel_left = array_vels[left_fwhm_idx]
    vel_right = array_vels[int(minimum_idx + right_fwhm_idx)]

    fwhm = np.abs(vel_right - vel_left)

    # FW20
    fw20_min = minimum / 5.

    left_fw20_idx = np.max(np.where(np.abs(left_array - fw20_min) ==
                                    np.abs(left_array - fw20_min).min())[0])
    right_fw20_idx = np.min(np.where(np.abs(right_array - fw20_min) ==
                                     np.abs(right_array - fw20_min).min())[0])

    vel_left = array_vels[left_fw20_idx]
    vel_right = array_vels[int(minimum_idx + right_fw20_idx)]

    fw20 = np.abs(vel_right - vel_left)

    return fwhm, fw20

# -------------------------------------------------#
#  FUNCTIONS of the MODEL                          #
#                                                  #
# - Load continuum image and interpolate           #
# to initial resolution                            #
# - Define the disk                                #
# - Compute integrated absorption line for each    #
# disk and merge them together                     #
#                                                  #
# ---------------                                  #
# Read Continuum                                   #
# -------------------------------------------------#


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
	    print "HERE"
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


# -------------------------------------------------#
# Function for the DISK coordinates and velocity   #
# -------------------------------------------------#

def space(z_cube_s, y_cube_s, x_cube_s, continuum_cube_s, rmax_s, rmin_s, h_0_s, pa_s, i_s):
    # -------------------------------------------------#
    # takes an input cube in space coordinates and the #
    # continuum cube returns the cube of velocities    #
    # of the disk and the continuum cube with values   #
    # only where there is absorption                  #
    # -------------------------------------------------#

    # trigonometric parameters of the disk
    i_rad = math.radians(i_s)
    pa_rad = math.radians(pa_s)

    # Disk
    cos_i = np.cos(i_rad)
    sin_i = np.sin(i_rad)
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)

    # convert into disk coordinates
    x = cos_pa * x_cube_s + sin_pa * y_cube_s
    y = cos_i * (-sin_pa * x_cube_s + cos_pa * y_cube_s) + sin_i * z_cube_s
    z = -sin_i * (-sin_pa * x_cube_s + cos_pa * y_cube_s) + cos_i * z_cube_s

    # determine the radius of the disk
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    angle = np.arctan2(y, x)

    # define the cube of velocities
    vel = -SIGN * sin_i * np.cos(angle) * VROT

    index_vel = (continuum_cube_s == 0.0)
    vel[index_vel] = -np.inf

    # condition for DISK 1
    idx = ((r > rmax_s) | (r < rmin_s) | (abs(z) >= h_0_s / 2.))
    # set to zero or -999 the flux and velocities outside the disk
    continuum_cube_s[idx] = 0.0
    vel[idx] = -np.inf

    # set to non good values what is behind the continuum
    # determine what is in front and what behind the continuum
    index_abs = (z_cube_s > (np.tan(PA_C_rad) * x_cube_s))
    # modify the cubes: bad values for what is behind
    vel[index_abs] = -np.inf
    continuum_cube_s[index_abs] = 0.0

    return vel, continuum_cube_s


# -------------------------------------------------#
# Function computing absorption for each disk     #
# -------------------------------------------------#

def mod_abs(z_cube_m, y_cube_m, x_cube_m, continuum_cube_z_m, flag, inc, pos_ang):
    continuum_cube_m = continuum_cube_z_m.copy()

    # -------------------------------------------------#
    # Load the parameters for the right disk          #
    # -------------------------------------------------#

    # flag=2 inner second disk
    if flag == int(2):
        rmax_m = float(par.get('rmax_in'))
        rmin_m = float(par.get('rmin_in'))
        h_0_m = float(par.get('h0_in'))
        i_m = I
        pa_m = PA
        # i=float(par.get('i_in'))
        # pa=float(par.get('pa_in'))

    # flag=1 outer first (or only) disk
    if flag == int(1):
        rmax_m = float(par.get('rmax'))
        rmin_m = float(par.get('rmin'))
        h_0_m = float(par.get('h0'))
        i_m = inc
        pa_m = pos_ang
        # i=float(par.get('i'))
        # pa=float(par.get('pa'))

    # -------------------------------------------------#
    # VELOCITY and FLUX of the absorbed disk          #
    # -------------------------------------------------#

    # compute the spectrum and the cubes
    velocity, flusso = space(z_cube_m, y_cube_m, x_cube_m, continuum_cube_m, rmax_m, rmin_m, h_0_m, pa_m, i_m)

    # -------------------------------------------------#
    # INTERPOLATE to final RESOLUTION                 #
    # -------------------------------------------------#

    print '...start interpolation...\n'

    # determine the factor for the interpolation depending on
    # the specified resolutions

    factor = RES / RES_FIN

    print '\tInterpolation of a factor:\t' + str(factor) + '\n'

    # increase the order for non-linear interpolation
    vel_zoom = zoom(velocity, factor, order=ORDER)
    flux_zoom = zoom(flusso, factor, order=ORDER)

    print '...end interpolation...\n'

    # -------------------------------------------------#
    # BIN the CUBE in the Integrated spectrum         #
    # -------------------------------------------------#

    print '...start binning...\n'

    # select velocities and fluxes which belong to the disk
    vel_index = ((vel_zoom >= -VROT) & (vel_zoom <= VROT))

    lin_vel = vel_zoom[vel_index]
    lin_flux = flux_zoom[vel_index]

    # determine the integrated spectrum
    spec = np.zeros([len(vels)])
    for k in xrange(0, len(vels) - 1):
        # look for the right velocity bin
        index = (vels[k] <= lin_vel) & (lin_vel < vels[k + 1])
        # update the flux bin
        spec[k] = -np.sum(lin_flux[index])

    print '...end binning...\n'

    return spec


# --------------------------------------------------#
# MCM functions                                    #
# --------------------------------------------------#


def chi_square(s_mod, s_obs):
    # interpolate to have arrays all of the same length
    func_obs = interp1d(s_obs[:, 0], s_obs[:, 1])
    func_mod = interp1d(s_mod[0, :], s_mod[1, :])

    # create a new array long enough
    vel_int = np.arange(-VROT, VROT + DISP, 2. * DISP)

    # determine the fluxes of observed
    # and modelled spectra

    obs_int = func_obs(vel_int)
    mod_int = func_mod(vel_int)

    res = obs_int - mod_int

    # set arrays for output
    res_out = np.array([vel_int, res])
    obs_out = np.array([vel_int, obs_int])
    mod_out = np.array([vel_int, mod_int])

    return res_out, mod_out, obs_out


# ----------------#


def lnprior(theta):
    inc, pos_ang = theta
    if I_d < inc < I_u and PA_d < pos_ang < PA_u:
        return 0.0
    return -np.inf


# ----------------#


def lnprob(theta, *par):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, continuum_cube_z, z_cube, y_cube, x_cube, spec_obs)


# ----------------#


def lnlike(theta, continuum_cube_z_ln, z_cube_ln, y_cube_ln, x_cube_ln, spec_obs_ln):
    # ** MAIN Loop **
    #     - create cube of coordinates
    #     - compute integrated spectrum
    #     - compute cubes of
    #         - absorbed disk
    #         - disk in front of continuum
    #         - disk behind continuum
    #     - convolve integrated spectrum
    #     - normalize spectrum to peak of observed spectrum
    #     - loop over mcmc algorithm

    inc, pos_ang = theta

    print inc, pos_ang
    # open output table

    print '********************\n'
    print '...Begin...\n'

    # -------------------------------------------------#
    # Compute absorption                               #
    # -------------------------------------------------#

    # tempoinit_modabs=time.time()

    print '...compute absorption for disk 1...\n'

    # compute the spectrum and the cubes
    spec_integral = mod_abs(z_cube_ln, y_cube_ln, x_cube_ln, continuum_cube_z_ln, int(1), inc, pos_ang)

    # set final integrated spectrum
    spec_int[1, :] = spec_integral[:]

    # -------------------------------------------------#
    # CONVOLVE the SPECTRUM                            #
    # -------------------------------------------------#

    # print '...convolve spectrum...\n'

    # convolve the spectrum at the desired velocity resolution
    # spec_int[1,:]=smoothing_func(spec_int[1,:])

    # -------------------------------------------------#
    # NORMALIZE the SPECTRUM to the OBSERVED one       #
    # -------------------------------------------------#

    print '...normalize spectrum...\n'

    # normalize modelled spectrum to observed one
    peak_obs = np.min(spec_obs_ln[:, 1])
    peak_mod = np.min(spec_int[1, :])

    if peak_mod < 0.:
        spec_int_norm = np.divide(spec_int[1, :], peak_mod)
        spec_int_mod = np.multiply(spec_int_norm, peak_obs)

    else:
        spec_int_mod = spec_int[1, :]

    # -------------------------------------------------#
    # Determine residuals & chi2 & widths of the line  #
    # -------------------------------------------------#

    print '...compute some stats...\n'

    spec_full = np.array([vels, spec_int_mod])

    res, mod_res, obs_res = chi_square(spec_full, spec_obs_ln)

    noise = np.std(obs_res[1, :])
    inv_noise = 1. / (noise * noise)

    loglike = -0.5 * np.sum((np.power(res[1, :], 2) * inv_noise - np.log(inv_noise)))
    print loglike

    FWHM, FW20 = widths(spec_full)

    # -------------------------------------------------#
    #write table
    RUN = 0
    with open(out_table, 'rb') as f:
        for line in f:
            RUN += 1

    print RUN

    table_out = open(out_table, "ab+")
    line_1 = str(RUN) + ''',''' + str(GAL) + ''',''' + str(RA) + ''',''' + str(DEC) + ''','''
    line_2 = str(z_red) + ''',''' + str(VSYS) + ''',''' + str(D_L) + ''',''' + str(PA_C) + ''','''
    line_3 = str(RMAX) + ''',''' + str(RMIN) + ''',''' + str(H_0) + ',' + str(inc) + ',' + str(pos_ang) + ''','''
    line_4 = str(RMAXD2) + ''',''' + str(RMIND2) + ''',''' + str(H_0D2) + ''',''' + str(ID2) + ''',''' + str(
        PAD2) + ''','''
    line_5 = str(VROT) + ''',''' + str(SIGN) + ''',''' + str(2 * DISP) + ''','''
    line_6 = str(RES) + ''',''' + str(RES_FIN) + ''',''' + str(loglike) + ''','''
    line_7 = str(FWHM) + ''',''' + str(FW20) +  '''\n'''

    line = line_1 + line_2 + line_3 + line_4 + line_5 + line_6 + line_7
    table_out.write(line)
    table_out.close()

    return loglike


# -------------------------------------------------#
# MAIN MAIN MAIN                                  #
# -------------------------------------------------#

tempoinit = time.time()

# -------------------------------------------------#
# SET the VARIABLES                                #
# -------------------------------------------------#

# Call the function and set the dictionary
par = readfile(fileinput)

# Disk 1
RMAX = float(par.get('rmax'))  # Maximum radius [pc]
RMIN = float(par.get('rmin'))  # Minimum radius [pc]
H_0 = float(par.get('h0'))  # Thickness [pc]
I = float(par.get('i'))  # Inclination [degrees]
PA = float(par.get('pa'))  # Position angle [degrees]
#convert coordinates : in np array 0 is along x axis
# in astro 0 is along North (y) axis
PA += 90.
par['pa'] = PA
# Disk 2
# !!! RMAXD2 < RMAX !!! #
RMAXD2 = float(par.get('rmax_in'))
RMIND2 = float(par.get('rmin_in'))
H_0D2 = float(par.get('h0_in'))
ID2 = float(par.get('i_in'))
PAD2 = float(par.get('pa_in'))

# Rotation Curve
VROT = float(par.get('vrot'))  # Flat limit of the rotation curve [km/s]
SIGN = float(par.get('sign'))  # Direction of rotation [=/- 1]

# Continuum information
PA_C = float(par.get('pa_cont'))  # Position angle of the continuum [degrees]
CONT_LIM = float(par.get('flux_cont_lim'))  # Noise limit: sets the region of the absorption !!!
VSYS = float(par.get('v_sys'))  # Systemic velocity [km/s]
D_L = float(par.get('d_l'))  # luminosity distance [Mpc]
z_red = float(par.get('z'))  # redshift
RA = par.get('ra')  # Right Ascention
DEC = par.get('dec')  # Declination
RA = string.strip(RA)
DEC = string.strip(DEC)

# Resolution Information
RES = float(par.get('pix_res'))  # Resolution 1st cycle  [pc]
RES_FIN = float(par.get('pix_res_fin'))  # Final resoluion   [pc]
VRES = float(par.get('vel_res'))  # Velocity resolution for binning [km/s]
DISP = float(par.get('disp'))  # Dispersion for final resolution (~ to observed spectrum) [km/s]
ORDER = int(par.get('order'))  # Order of spline for interpolation

# MCMC parameters
DIM_mcmc = int(par.get('ndim_mcmc'))
WALK_mcmc = int(par.get('nwalkers_mcmc'))
STEPS_MCMC = int(par.get('nsteps_mcmc'))

I_d = float(par.get('I_left'))
I_u = float(par.get('I_right'))

PA_d = float(par.get('PA_left'))
PA_u = float(par.get('PA_right'))
#convert coordinates : in np array 0 is along x axis
# in astro 0 is along North (y) axis 
PA_d = PA_d + 90. 
PA_u = PA_u + 90.
par['pa_left'] = PA_d
par['pa_right'] = PA_u

# galaxy name in the parameter dictionary
par['gal'] = GAL

# -------------------------------------------------#
# Print variables                                  #
# -------------------------------------------------#

print '********************\n'
print 'INPUT PARAMETERS\n'
print '********************\n'
for keys, values in par.items():
    print keys + ' = ' + str(values)
print '********************\n'

# -------------------------------------------------#
# Convert coordinates                              #
# -------------------------------------------------#

# convert to degrees
ra = ra2deg(RA)
dec = dec2deg(DEC)
# convert degrees to radians
PA_C_rad = math.radians(PA_C)

# -------------------------------------------------#
# CUBE                                             #
# -------------------------------------------------#

print 'INPUT DISK\n'
print '********************\n'

# build the cube which contains my disk based on its (RMAX)
# and on the resolution of the 1st cycle

# set the edges to RMAX+20%
x_los = np.arange(-RMAX * 1.05, +RMAX * 1.05 + RES, RES)
y_los = np.arange(-RMAX * 1.05, +RMAX * 1.05 + RES, RES)
z_los = np.arange(-RMAX * 1.05, +RMAX * 1.05 + RES, RES)

# edges of the cube
print 'edges of the cube [pc]:\t\t\t' + str(x_los[-1]), str(x_los[0])
print 'size of the cube [pixels]]:\t\t' + str(len(x_los)) + ' x ' + str(len(y_los)) + ' x ' + str(len(z_los))
print 'resolution first cycle [pc]:\t\t' + str(RES) + '\n'

# -------------------------------------------------#
# INTEGRATED spectrum                              #
# -------------------------------------------------#
#                                                 #
# input integrated spectrum:                       #
# velocity = bins of desired resolution            #
# define the output integrated spectrum            #
# one bin must include zero (zero not an edge!!!)  #
#                                                 #
# -------------------------------------------------#


# edges of the velocity array
# print 'edges of the velocity array [km/s]:\t' + str(vels[-1]),str(vels[0])
print 'velocity resolution [km/s]:\t\t' + str(VRES)
print 'convolved velocity resolution [km/s]:\t' + str(2. * DISP) + '\n'
print '********************\n'

# -------------------------------------------------#
# Set the input continuum cube                     #
# -------------------------------------------------#

print '... set continuum input cube...\n'

# load the continuum image interpolated at the specified resolution
continuum_image = build_continuum(y_los, x_los)

# -------------------------------------------------#
# create a cube containing the fluxes of           #
# the continuum image                             #
# based on the coordinate system of the README    #
# the axis are sorted in the array: [y,z,x]       #
# -------------------------------------------------#

continuum_cube_z = np.dstack([continuum_image] * (len(z_los)))
continuum_cube_z = np.swapaxes(continuum_cube_z, 1, 2)

# mask the noise of the continuum in the cube
index_mask = continuum_cube_z < CONT_LIM
continuum_cube_z[index_mask] = 0.0
continuum_cube_z2 = continuum_cube_z.copy()

# mask the noise of the continuum in the continuum image
index_mask = continuum_image < CONT_LIM
continuum_image[index_mask] = 0.0

# -------------------------------------------------#
# Set the input coordinates cube                  #
# -------------------------------------------------#

print '...set disk...\n'

# create a meshgrid for the cube
z_cube, y_cube, x_cube = np.meshgrid(z_los, y_los, x_los)

# -------------------------------------------------#
# load observed spectrum
spec_obs = np.loadtxt(filespec)
# convert in frequency given the systemic velocity
spec_obs[:, 0] = (HI_hz / spec_obs[:, 0] - 1) * C
spec_obs[:, 0] = spec_obs[:, 0] - VSYS

vels = np.hstack([1.5 * spec_obs[0, 0] - 0.5 * spec_obs[1, 0],
                  0.5 * (spec_obs[0:-1, 0] + spec_obs[1:, 0]),
                  1.5 * spec_obs[-1, 0] - 0.5 * spec_obs[-2, 0]])

if vels[-1]<vels[0]:
        vels=vels[::-1]
	spec_obs[:,0]= spec_obs[::-1,0]
spec_int = np.zeros([2, len(vels)])
spec_int[0, :] = vels[:]

print spec_obs,spec_int
print '...spectrum loaded..'

# -------------------------------------------------#
#load / create table
if os.path.exists(out_table) is True:
    table_out = open(out_table, "ab+")
else:
    table_out = open(out_table, "ab+")
    # write title line if table does not exist
    title_line_1 = '''#RUN,GAL,RA,DEC,z,v_sys[kms],D_L[Mpc],PA_cont[degrees],'''
    title_line_2 = '''R_max[pc],R_min[pc],H[pc],I[degrees],PA[degrees]'''
    title_line_3 = '''R_maxD2[pc],R_minD2[pc],H_D2[pc],I_D2[degrees],PA_D2[degrees],'''
    title_line_4 = '''v_rot[kms],sign[-],v_res[kms],'''
    title_line_5 = '''res_in[pc],res_fin[pc],chi_square,'''
    title_line_6 = '''FWHM[kms],FW20[kms]\n'''


    title_line = title_line_1 + title_line_2 + title_line_3 + title_line_4 + title_line_5 + title_line_6
    table_out.write(title_line)

# -------------------------------------------------#
#start mcmc
p0 = []

for i in xrange(WALK_mcmc):
    inclinations = random.uniform(I_d, I_u)
    positionangles = random.uniform(PA_d, PA_u)

    p0.append((inclinations, positionangles))

print '... starting seeds ...'
print p0

# run mcmc algorithm
sampler = emcee.EnsembleSampler(WALK_mcmc, DIM_mcmc, lnprob,
                                args=(continuum_cube_z, z_cube, y_cube, x_cube, spec_obs, par),
                                threads=15)
sampler.run_mcmc(p0, STEPS_MCMC)

# save run outputs
samples = sampler.chain[:, :].reshape((-1, DIM_mcmc))
np.savetxt(out_samples_text, samples)
samp_chain = np.array(sampler.chain[:, :, :], dtype=float)
np.save(out_walkers_vec, samp_chain)


tempofin = (time.time() - tempoinit) / 60.
tempoinit = tempofin

print '********************'
print "\tTotal time: %f minutes" % tempofin

print '********************'
print 'NORMAL TERMINATION'
print '********************\n'
