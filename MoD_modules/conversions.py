import string
import numpy as np


RAD2DEG = 180./np.pi

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