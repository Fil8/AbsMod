
# coding: utf-8

# # MoD AbS

# In[315]:

##################################################
# #                 MoD AbS                    # #
# #     modelling disks for HI absorption      # #
# #      fit the line and writes a table       # #
#                                                #
# Filippo Maccagni 24 - 06 - 2016                #
# ASTRON - Kapteyn Institute                     #
##################################################

# import installed python modules
#!/usr/bin/env python
import math
import time
import numpy as np
import string,sys,os
import argparse
#import pyfits
import itertools
from astropy import wcs
from astropy.io import fits
from scipy.ndimage.interpolation import zoom
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from pylab import contour,cm,clabel

#load magic python notebook module
#to comment if run from terminal
#get_ipython().magic(u'pylab inline')

print'Modules Imported'

###################################################



# ** PATHS **
#     - Root directory
#     - Output directory (created if non-existing)
#     - Path to continuum image
#     - Path to spectrum

# In[316]:

#-------------------------------------------------#
#Directories                                      # 
#-------------------------------------------------#

#Galaxy Name
GAL='3C305'

#Root Directory
#directory where the program is located
rootdir=os.path.abspath('')

#Directory for the input files
root_obs=rootdir+'/3c305/'

#Directory for the output files
root_out=rootdir+'/output_3c305/'
try:
    os.stat(root_out)
except:
    os.mkdir(root_out)

#-------------------------------------------------#
#Output table                                     #
#-------------------------------------------------#

#fileinput=sys.argv[1]
#uncomment if loading from terminal#

out_table = rootdir+'/table_out_'+GAL+'.csv'
    
#-------------------------------------------------#
#Parameter file                                   #
#-------------------------------------------------#

#fileinput=sys.argv[1]
#uncomment if loading from terminal#

fileinput = root_obs+'par_3c305.txt'
 

#-------------------------------------------------#
#Continuum .fits file                             #
#-------------------------------------------------#

#filecont=sys.argv[2]

filecont = root_obs+'3c305.fits'

#-------------------------------------------------#
#Observed spectrum ASCII file                     #
#-------------------------------------------------#

#filespec=sys.argv[3]

filespec = root_obs+'spec_3c305.txt' #spectrum x-axis in Hz

#-------------------------------------------------#
#CONSTANTS                                        #
#-------------------------------------------------#

RAD2DEG=180./math.pi
HI_hz=1.42040575177e9
C=2.99792458E5 

###################################################


# ** Basic Functions **
#     - Read parameter file
#     - Write output log table
#     - Convert HMS -> degrees
#     - Convert DMS -> degrees
#     - Convolve with a Gaussian

# In[317]:



#-------------------------------------------------#
#Write the output table                           #
#-------------------------------------------------#

def writeTable(self,out_table):
    
    
    #open output table 
    if os.path.exists(out_table) == True: 
        table_out=open(out_table, "aw")
    else:
        table_out=open(out_table, "aw")
        #write title line if table does not exist
        title_line_1='''#RUN,GAL,RA,DEC,z,v_sys[kms],D_L[Mpc],PA_cont[degrees],'''
        title_line_2='''R_max[pc],R_min[pc],H[pc],I[degrees],PA[degrees],'''
        title_line_3='''R_maxD2[pc],R_minD2[pc],H_D2[pc],I_D2[degrees],PA_D2[degrees],'''
        title_line_4='''v_rot[kms],sign[-],v_res[kms],'''
        title_line_5='''res_in[pc],res_fin[pc],med_res,disp_res\n'''

        title_line=title_line_1+title_line_2+title_line_3+title_line_4+title_line_5
        table_out.write(title_line)
        
    
        
    line_1=str(self.RUN)+''','''+str(self.GAL)+''','''+str(self.RA)+''','''+str(self.DEC)+''','''
    line_11=str(self.z_red)+''','''+str(self.VSYS)+''','''+str(self.D_L)+''','''+str(self.PA_C)+''','''   
    line_2=str(self.RMAX)+''','''+str(self.RMIN)+''','''+str(self.H_0)+','+str(self.I)+','+str(self.PA)+''','''
    line_3=str(self.RMAXD2)+''','''+str(self.RMIND2)+''','''+str(self.H_0D2)+''','''+str(self.ID2)+''','''+str(self.PAD2)+''','''  
    line_4=str(self.VROT)+''','''+str(self.SIGN)+''','''+str(2*self.DISP)+''',''' 
    line_5=str(self.RES)+''','''+str(self.RES_FIN)+''','''+str(self.MED_RES)+''','''+str(self.NOISE_RES)+'''\n'''

    
    line=line_1+line_11+line_2+line_3+line_4+line_5
    table_out.write(line) 
    
    table_out.close()

    return 0

#-------------------------------------------------#
#Coordinates converter                            #
#-------------------------------------------------#

# HMS -> degrees
def ra2deg(rad):
    
        ra=string.split(rad,':')
    
        hh=float(ra[0])*15
        mm=(float(ra[1])/60)*15
        ss=(float(ra[2])/3600)*15
        
        return hh+mm+ss

# DMS -> degrees
def dec2deg(decd):
        dec=string.split(decd,':')
        
        hh=abs(float(dec[0]))
        mm=float(dec[1])/60
        ss=float(dec[2])/3600
        return hh+mm+ss

#-------------------------------------------------#
# Size converter                                  #
#-------------------------------------------------#

# Angle -> radius [Mpc]
def ang2lin(self,z,dl,ang): # r in arcsec   
    #dl = dl/3.085678e24 # Mpc
    r = ang * dl / (RAD2DEG * 3600 * (1+z)**2) # Mpc
    
    return r
# radius -> angle [arcsec]
def lin2ang(self,z,dl,r): # r in Mpc

    ang = RAD2DEG * 3600. * r * (1.+z)**2 / dl # arcsec

    return ang

#-------------------------------------------------#
# Gaussian Convolution                            #
#-------------------------------------------------#

def convoluzion(self,convo):
        mu=0.0        
        arg=-((vels*vels)/(2*self.DISP*self.DISP))
        gauss=1./(np.sqrt(2*np.pi)*self.DISP)*np.exp(arg)
        convolved_pdf=np.convolve(convo,gauss,mode='same')
         
        return convolved_pdf
    
###################################################



# In[318]:

#-------------------------------------------------#
#SET the VARIABLES                                #
#-------------------------------------------------#
class variables:

    #-------------------------------------------------#
    #Read the parameter file and create a dictionary  #
    #-------------------------------------------------#

    
    def set_variables(self):

        #Call the function and set the dictionary
        #par=readFile(fileinput)
        par={}
    
        try:
                paramFile=open(fileinput)
        except:
                print "%s file not found"  % fileinput
                return 1
        
        paramList=paramFile.readlines()
        paramFile.close()     
        for line in paramList:
                if line.strip():  # non-empty line?
                    tmp=line.split('=')
                    tmp2=tmp[0].split('[')
                    key, value = tmp2[0],tmp[-1]  # None means 'all whitespace', the default
                    par[key] = value
                    
                    
        # Disk 1 
        self.RMAX=float(par.get('rmax'))                   #Maximum radius [pc]
        self.RMIN=float(par.get('rmin'))                   #Minimum radius [pc]
        self.H_0=float(par.get('h0'))                      #Thickness [pc]
        self.I=float(par.get('i'))                         #Inclination [degrees]
        self.PA=float(par.get('pa'))                       #Position angle [degrees]

        #Disk 2 
        # !!! RMAXD2 < RMAX !!! #
        self.RMAXD2=float(par.get('rmax_in'))
        self.RMIND2=float(par.get('rmin_in'))
        self.H_0D2=float(par.get('h0_in'))
        self.ID2=float(par.get('i_in'))
        self.PAD2=float(par.get('pa_in'))

        # Rotation Curve
        self.VROT=float(par.get('vrot'))                   #Flat limit of the rotation curve [km/s]
        self.SIGN=float(par.get('sign'))                   #Direction of rotation [=/- 1]

        # Continuum information
        self.PA_C=float(par.get('pa_cont'))                #Position angle of the continuum [degrees]
        self.CONT_LIM=float(par.get('flux_cont_lim'))      #Noise limit: sets the region of the absorption !!!
        self.VSYS=float(par.get('v_sys'))                  #Systemic velocity [km/s]
        self.D_L=float(par.get('d_l'))                     #luminosity distance [Mpc]
        self.z_red=float(par.get('z'))                     #redshift
        self.RA=par.get('ra')                              #Right Ascention 
        self.DEC=par.get('dec')                            #Declination

        self.RA=string.strip(self.RA)
        self.DEC=string.strip(self.DEC)

        # Resolution Information
        self.RES=float(par.get('pix_res'))                 #Resolution 1st cycle  [pc]
        self.RES_FIN=float(par.get('pix_res_fin'))         #Final resoluion   [pc]
        self.VRES=float(par.get('vel_res'))                #Velocity resolution for binning [km/s]
        self.DISP=float(par.get('disp'))                   #Dispertion for final resolution (~ to observed spectrum) [km/s]
        self.ORDER=int(par.get('order'))                   #Order of spline for interpolation
        self.STEP=float(par.get('step'))                   #step with which we vary the parameters
        
        #galaxy name in the parameter dictionary
        par['gal'] = GAL
        self.GAL = GAL
        #-------------------------------------------------#
        #Print variables                                  #
        #-------------------------------------------------#

        print '********************\n'
        print 'INPUT PARAMETERS\n'
        print '********************\n'
        for keys,values in par.items():
            print keys+' = '+str(values)
        print '********************\n'   

        #-------------------------------------------------#
        #Convert coordinates                              #
        #-------------------------------------------------#

        #convert to degrees
        self.ra=ra2deg(self.RA)
        self.dec=dec2deg(self.DEC)
                
        #-------------------------------------------------#
        #Define limit arrays for ITERATION                #
        #-------------------------------------------------#
        
        self.I_LIM=[20.,70.]
        self.PA_LIM=[-60.,60.]
        self.PA_C_LIM=[-30.,70.]
        
        return self
    
    #-------------------------------------------------#
    #Read the PARAMETER FILE                          #
    #-------------------------------------------------#
    def readFile(parameterFile):
        
            parameters={}
    
            try:
                paramFile=open(parameterFile)
            except:
                print "%s file not found"  % parameterFile
                return 1
        
            paramList=paramFile.readlines()
            paramFile.close()     
            print paramList
            print '**********'
            for line in paramList:
                if line.strip():  # non-empty line?
                    tmp=line.split('=')
                    tmp2=tmp[0].split('[')
                    key, value = tmp2[0],tmp[-1]  # None means 'all whitespace', the default
                    parameters[key] = value
    
            return parameters
            
#-------------------------------------------------#
#Call class VARIABLES                             #
#-------------------------------------------------#   

v=variables()
#define the self set of variables
self=v.set_variables()


print 'NORMAL TERMINATION'


# ** FUNCTIONS of the MODEL **
# 
#     - Load continuum image and interpolate to initial resolution
#     - Define the disk
#     - Compute integrated absorption line for each disk and merge them together
#     - Determine residuals of modelled spectrum w.r.t. observed one

# In[319]:

#-------------------------------------------------#
#Continuum CUBE: Input for the absorption        #
#-------------------------------------------------#

def build_continuum(self,x_los,y_los):
        
        #Load continuum: 
        f=fits.open(filecont)
        dati=f[0].data
        head=f[0].header
        dati=np.squeeze(dati)
        dati=np.squeeze(dati)

        # define the resolution of the continuum image
        scale_cont_asec=head['CDELT2']*3600
        scale_cont_pc=ang2lin(self,self.z_red,self.D_L,scale_cont_asec)*1e6


        #load the continuum image
        head=fits.getheader(filecont)
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


        w=wcs.WCS(head)    
        
        #convert coordinates in pixels
        #cen_x,cen_y=w.wcs_world2pix(ra,dec,0)
        cen_x,cen_y=w.wcs_world2pix(self.ra,self.dec,1)
        
        print '\tContinuum centre [pixel]:\t'+'x: '+str(cen_x)+'\ty: '+str(cen_y) 
        print '\tContinuum pixel size [pc]:\t'+str(scale_cont_pc)+'\n'
        
        
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
        x_left_int=math.modf(x_left)
        x_right_int=math.modf(x_right)
        y_low_int=math.modf(y_low)
        y_up_int=math.modf(y_up)
        
        #select the continuum subset
        sub_dati=dati[int(y_low_int[1]):int(y_up_int[1]),int(x_left_int[1]):int(x_right_int[1])]

        if sub_dati.shape[0] != sub_dati.shape[1]:
            sub_dati=dati[int(y_low_int[1]):int(y_up_int[1])-1,int(x_left_int[1]):int(x_right_int[1])]
            
        
        #determine how much I have to interpolate     
        zoom_factor= float(len(x_los))/float(len(sub_dati[0]))
        
        #interpolate to the desired resolution of the cycle 1 cube
        zoom_dati=zoom(sub_dati,zoom_factor,order=3)
                               
        return zoom_dati
    
#-------------------------------------------------#
#Function for the DISK coordinates and velocity   #
#-------------------------------------------------#

def space(self,z_cube,y_cube,x_cube,continuum_cube):  
    
#-------------------------------------------------#
#takes an input cube in space coordinates and the # 
#continuum cube returns the cube of velocities    #
#of the disk and the continuum cube with values   # 
# only where there is absorption                  #
#-------------------------------------------------#


        #trigonometric parameters of the disk
        i_rad=math.radians(self.i)
        pa_rad=math.radians(self.pa)
 
        self.PA_C_rad=math.radians(self.PA_C)

        #Disk
        cos_i = np.cos(i_rad)
        sin_i = np.sin(i_rad)
        cos_pa = np.cos(pa_rad)
        sin_pa = np.sin(pa_rad)

        #convert into disk coordinates
        x=cos_pa*x_cube+sin_pa*y_cube
        y=cos_i*(-sin_pa*x_cube+cos_pa*y_cube)+sin_i*z_cube  
        z=-sin_i*(-sin_pa*x_cube+cos_pa*y_cube)+cos_i*z_cube
         

        #determine the radius of the disk    
        r=np.sqrt(np.power(x,2)+np.power(y,2))
        angle=np.arctan2(y,x)        
       
        #define the cube of velocities 
        vel=-self.SIGN*sin_i*np.cos(angle)*self.VROT
        
        #for plotting: define the disk in front of the continuum
        disk_front=vel.copy()
        
        index_vel= (continuum_cube == 0.0)
        vel[index_vel]=-np.inf
        
        #condition for DISK 1
        idx = ( (r > self.rmax) |  (r<self.rmin) | (abs(z)>=self.h_0/2.) )
        #set to zero or -999 the flux and velocities outside the disk
        continuum_cube[idx] = 0.0
        vel[idx]= -np.inf
        disk_front[idx]= -999.

        
        #set to non good values what is behind the continuum
        #for plotting: define the disk behind the continuum
        disk_behind=disk_front.copy()
        
        #determine what is in front and what behind the continuum
        index_abs = (z_cube>(np.tan(self.PA_C_rad)*x_cube))
        
        #modify the cubes: bad values for what is behind
        vel[index_abs] = -np.inf        
        continuum_cube[index_abs] = 0.0
        disk_front[index_abs]=-999.

        index_abs = (z_cube<(np.tan(self.PA_C_rad)*x_cube))
        
        disk_behind[index_abs]=-999.
        
        #for plotting: exclude the absorbed section from the disk in front
        disk_ind= ((vel >= -self.VROT) & (vel <= self.VROT) )
        disk_front[disk_ind]= -999.
        
        return vel,continuum_cube,disk_front,disk_behind


#-------------------------------------------------#
#Function computing absorption for each disk     #  
#-------------------------------------------------#

def mod_abs(self,z_cube,y_cube,x_cube,continuum_cube_z,flag):
 
    #-------------------------------------------------#
    # Load the parameters for the right disk          #
    #-------------------------------------------------#       
    
    #flag=1 outer first (or only) disk
    if flag == int(1):
        self.rmax=self.RMAX
        self.rmin=self.RMIN
        self.h_0=self.H_0
        self.pa=self.PA
        self.i=self.I
    
    #flag=2 inner second disk
    if flag == int(2):
        self.rmax=self.RMAXD2
        self.rmin=self.RMIND2
        self.h_0=self.H_0D2    
        self.pa=self.PA_D2
        self.i=self.I_D2

        
        
    #-------------------------------------------------#
    # VELOCITY and FLUX of the absorbed disk          #
    #-------------------------------------------------#                    
    
    velocity,flusso,disk_front,disk_behind = space(self,z_cube,y_cube,x_cube,continuum_cube_z)
    
    #-------------------------------------------------#
    # INTERPOLATE to final RESOLUTION                 #
    #-------------------------------------------------#
    
    print '...start interpolation...\n'

    #determine the factor for the interpolation depending on 
    #the specified resolutions
    
    factor=self.RES/self.RES_FIN

    print '\tInterpolation of a factor:\t'+str(factor)+'\n'

    #increase the order for non-linear interpolation
    vel_zoom = zoom(velocity, factor,order=self.ORDER)
    flux_zoom = zoom(flusso, factor, order=self.ORDER)

    print '...end interpolation...\n'

    
    #-------------------------------------------------#
    # BIN the CUBE in the Integrated spectrum         #
    #-------------------------------------------------#
    
    print '...start binning...\n'
    
    #select velocities and fluxes which belong to the disk
    vel_index= ((vel_zoom >= -self.VROT) & (vel_zoom <= self.VROT) )

    #straighten the array
    lin_vel=vel_zoom[vel_index]
    lin_flux=flux_zoom[vel_index]
    
    #determine the integrated spectrum
    spec=np.zeros([len(vels)])
    for i in xrange(0,len(vels)-1):
        #look for the right velocity bin
        index=(vels[i]<=lin_vel) & (lin_vel < vels[i+1])
        #update the flux bin
        spec[i]=-np.sum(lin_flux[index])
    
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
    
    return spec,velocity,disk_front,disk_behind


#-------------------------------------------------#
#CHI2 and RESIDUAL spectrum                       #
#-------------------------------------------------#
def chi_res(self,s_mod,s_obs):
    
    #interpolate to have arrays all of the same length
    func_obs= interp1d(s_obs[:,0],s_obs[:,1])
    func_mod= interp1d(s_mod[0,:],s_mod[1,:])
    
    #create a new array long enough
    vel_int=np.arange(-self.VROT,self.VROT+self.DISP,2.*self.DISP)
    
    #determine the fluxes of observed 
    #and modelled spectra
    
    obs_int=func_obs(vel_int)
    mod_int=func_mod(vel_int)

    #determine residuals array
    res=obs_int-mod_int

    #set arrays for output
    res_out=np.array([vel_int,res])
    obs_out=np.array([vel_int,obs_int])
    mod_out=np.array([vel_int,mod_int])


    return res_out,obs_out,mod_out
    

###################################################


# ** PLOT SPECTRUM and DISK **

# In[320]:

#-------------------------------------------------#
# PLOT function                                   #
#-------------------------------------------------#

def plot_figure(self,spec_obs,spec_int,res,mod,obs,outfile_fig,flusso,continuum_image,disk_front,disk_behind):
  
    #set limits of images
    #spectrum limits [km/s]
    xleft=-self.VROT*3
    xright=self.VROT*3
    
    yup=np.max(spec_obs[:,1]*1.05)
    ydown=np.min(spec_obs[:,1]*1.05)

    #enlarge limits of the modelled spectrum
    vels_enl=np.arange(xleft,xright+self.VRES,self.VRES)
    spec_enl=np.zeros(len(vels_enl))
    
    left=(len(vels_enl)-len(vels))/2
    right=len(vels_enl)-left
    
    spec_enl[left:right]=spec_int[:]
    spec_enl[left-1]=spec_int[0]/2.
    spec_enl[right+1]=spec_int[-1]/2.
    
    #define figure parameters  
    params = {'legend.fontsize': 14,
           'axes.linewidth':2,
            'axes.labelsize':22,
           'lines.linewidth':1,
           'legend.linewidth': 3,
           'xtick.labelsize':22,
           'ytick.labelsize':22,
           'xlabel.fontsize':22,
           'ylabel.fontsize':22,
           'text.usetex': True,
           'text.latex.unicode' : True }
    rc('font',**{'family':'serif','serif':['serif']})
    plt.rcParams.update(params)
          

    #-------------------------------------------------#
    # Set FIGURE and GRID                             #
    #-------------------------------------------------#
    
    fig_a = plt.figure(figsize=(18, 18), dpi=100)

    #set the FULL grid
    gs_all = gridspec.GridSpec(2, 1)
    gs_all.update(left=0.1, right=0.9, wspace=0.0,hspace=0.00)
    
    #set the grid for the spectrum and the projected cube
    gs_spec = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_all[0], wspace=0.0, hspace=0.0)
    gs_ort = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_all[1], wspace=0.2, hspace=0.0)

    
    #-------------------------------------------------#
    # Plot the spectrum and parameters of the disk    #
    #-------------------------------------------------#
    
    #define plots
    ax_spec=fig_a.add_subplot(gs_spec[0:2, 0:2])
    ax_par=fig_a.add_subplot(gs_spec[0:3, 2])
    ax_res=fig_a.add_subplot(gs_spec[2, 0:2])

    
    #-------------------------------------------------#
    # Spectrum                                        #
    #-------------------------------------------------#  
    
    #plot the observed and modelled spectra
    ax_spec.plot(spec_obs[:,0],spec_obs[:,1],ls='-',c='black',label=r'WSRT',marker=' ',lw=3)
    ax_spec.plot(vels_enl,spec_enl,ls='-',c='red',label=r'model',marker=' ',lw=2)

    ax_spec.plot(obs[0,:],obs[1,:],ls='-',c='blue',marker=' ',lw=2)

    
    #plot residuals and observed spectrum
    ax_res.plot(spec_obs[:,0],spec_obs[:,1],ls='-',c='black',label=r'WSRT',marker=' ',lw=3)
    ax_res.plot(res[0,:],res[1,:],ls='-',c='orange',label=r'residuals',marker=' ',lw=3)

    
    #plot horizontal line at zero
    xx=[xleft,xright]
    yy=[0.,0.]
    ax_spec.plot(xx,yy,ls='--',lw=1,color='black')
    ax_res.plot(xx,yy,ls='--',lw=1,color='black')

    #plot vertical line at zero
    xx=[0.,0.]
    yy=[ydown,yup]
    ax_spec.plot(xx,yy,ls='--',lw=1,color='black')
    ax_res.plot(xx,yy,ls='--',lw=1,color='black')

    #plot vertical line at edges of rotation curve
    xx=[res[0,0],res[0,0]]
    yy=[ydown,yup]
    ax_res.plot(xx,yy,ls='--',lw=1,color='black')   
    xx=[res[0,0-1],res[0,-1]]
    yy=[ydown,yup]
    ax_res.plot(xx,yy,ls='--',lw=1,color='black')      
    
    #set limits of image
    ax_spec.set_xlim(xleft,xright)
    ax_spec.set_ylim(ydown,yup)

    ax_res.set_xlim(xleft,xright)
    ax_res.set_ylim(ydown,yup)

    
    #set labesl of spectrum
    ax_spec.set_ylabel(r'Flux\, [Jy]')
    ax_spec.legend(loc=4)

    ax_res.legend(loc=4)

    ax_res.set_xlabel(r'Velocity [km\,s$^{-1}$]')

    #set title
    ax_spec.set_title('Spectrum',fontsize=22)


    #-------------------------------------------------#
    # Disk parameters                                 #
    #-------------------------------------------------#    
    
    #plot box with parameters
    ax_par.text(0.1, 0.94, r' r$_{\rm in}$\,(max)\,\,=\,'+str(self.RMAX)+r'\,\,pc', fontsize=14)
    ax_par.text(0.1, 0.87, r' r$_{\rm in}$\,(min)\,\,=\,'+str(self.RMIN)+r'\,\,pc', fontsize=14)
    ax_par.text(0.1, 0.80, r' h$_{\rm in}$\,\,\,=\,'+str(self.H_0)+r'\,\,pc', fontsize=14)
    ax_par.text(0.1, 0.73, r' pa$_{\rm in}$\,\,=\,'+str(self.PA)+r'\,\,$^\circ$', fontsize=14)
    ax_par.text(0.1, 0.66, r' i$_{\rm in}$\,\,=\,'+str(self.I)+r'\,\,$^\circ$', fontsize=14)
    ax_par.text(0.1, 0.59, r' r$_{\rm out}$\,(max)\,\,=\,'+str(self.RMAXD2)+r'\,\,pc', fontsize=14)
    ax_par.text(0.1, 0.52, r' r$_{\rm out}$\,(min)\,\,=\,'+str(self.RMIND2)+r'\,\,pc', fontsize=14)
    ax_par.text(0.1, 0.45, r' h$_{\rm out}$\,\,\,=\,'+str(self.H_0D2)+r'\,\,pc', fontsize=14)
    ax_par.text(0.1, 0.38, r' pa$_{\rm out}$\,\,=\,'+str(self.PAD2)+r'\,\,$^\circ$', fontsize=14)
    ax_par.text(0.1, 0.31, r' i$_{\rm out}$\,\,=\,'+str(self.ID2)+r'\,\,$^\circ$', fontsize=14)
    ax_par.text(0.1, 0.24, r' v\,(rot)\,\,=\,'+str(self.VROT)+r'\,\,km\,s$^{-1}$', fontsize=14)
    ax_par.text(0.1, 0.17, r' pa \,(cont)\,\,=\,'+str(self.PA_C)+r'\,\,$^\circ$', fontsize=14)

    ax_par.text(0.1, 0.10, r' med \,res\,\,=\,'+str(round(self.MED_RES,5))+r'\,\,Jy', fontsize=14)
    ax_par.text(0.1, 0.03, r' disp \,res\,\,=\,'+str(round(self.NOISE_RES,5))+r'\,\,Jy', fontsize=14)

    #ax_par.text(0.1, 0.2, r' RA\,\,=\,'+RA, fontsize=16)
    #ax_par.text(0.1, 0.1, r' DEC \,\,=\,'+DEC, fontsize=16)
    
    #set ticks
    ax_par.set_xticks([])
    ax_par.set_yticks([])
    #set title
    ax_par.set_title('Disk parameters',fontsize=22)
 

    #-------------------------------------------------#
    # Plot the cube in different projections          #
    #-------------------------------------------------#
    
    #define plots
    ax_pv = fig_a.add_subplot(gs_ort[0,0]) 
    ax_po = fig_a.add_subplot(gs_ort[0,1])
    ax_pl = fig_a.add_subplot(gs_ort[0,2])
    
    
    #-------------------------------------------------#
    # PLANE of the SKY                                #
    #-------------------------------------------------#
    
    #project the fluxes of the absorbed cube
    flux_zoom_pv=np.nanmean(flusso,axis=1)
    disk_front_pv=np.nanmean(disk_front,axis=1)
    disk_behind_pv=np.nanmean(disk_behind,axis=1)
    
    #plot continuum image
    ax_pv.imshow(continuum_image,extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]],origin='lower',cmap='hot_r',alpha=0.8)
    cont=[self.CONT_LIM]
    ax_pv.contour(continuum_image,cont,origin='lower',
                     colors='black',linewidths=3,ls='-.',extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]])

    #plot absorbed part of the disk
    ax_pv.imshow(flux_zoom_pv,origin='lower',cmap='nipy_spectral',extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]],alpha=1.)   

    #plot disk in front of continuum
    ax_pv.imshow(disk_front_pv,origin='lower',cmap='nipy_spectral',extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]],alpha=0.1)   
    #plot disk in behind continuum
    ax_pv.imshow(disk_behind_pv,origin='lower',cmap='nipy_spectral',extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]],alpha=0.4)   
    
    #set ticks
    ax_pv.set_xlabel(r'x [kpc]')
    ax_pv.set_ylabel(r'y [kpc]')
    ax_pv.set(adjustable='box-forced', aspect='equal')
    #ax_pv.set_xticklabels([-4,-3,-2,-1,0,1,2,3])
    #ax_pv.set_yticklabels([-4,-3,-2,-1,0,1,2,3])    
    #set title
    ax_pv.set_title('Plane of the sky',fontsize=22)

    
    #-------------------------------------------------#
    # View from above                                 #
    #-------------------------------------------------#

    #project the fluxes of the absorbed cube
    flux_zoom_po=np.nanmean(flusso,axis=0)
    disk_front_po=np.nanmean(disk_front,axis=0)
    disk_behind_po=np.nanmean(disk_behind,axis=0)

    #plot absorbed part of the disk
    ax_po.imshow(flux_zoom_po,origin='lower',cmap='nipy_spectral',extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]],alpha=1.)     
    
    #plot disk in front of continuum
    ax_po.imshow(disk_front_po,origin='lower',cmap='nipy_spectral',extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]],alpha=0.1)   
    #plot disk in behind continuum
    ax_po.imshow(disk_behind_po,origin='lower',cmap='nipy_spectral',extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]],alpha=0.4)   

    #set ticks
    ax_po.set_xlabel(r'x [kpc]')
    ax_po.set_ylabel(r'z [kpc]')
    ax_po.set(adjustable='box-forced', aspect='equal')
    #ax_po.set_yticks([])
    #ax_po.set_xticklabels([-4,-3,-2,-1,0,1,2,3])
    #ax_po.set_yticklabels([-4,-3,-2,-1,0,1,2,3])
    
    #set title
    ax_po.set_title('View from `above`',fontsize=22)   

    
    #-------------------------------------------------#
    # View from the side                              #
    #-------------------------------------------------#
    
    #project the fluxes of the absorbed cube
    flux_zoom_pl=np.nanmean(flusso,axis=2)
    disk_front_pl=np.nanmean(disk_front,axis=2)
    disk_behind_pl=np.nanmean(disk_behind,axis=2)

    #plot absorbed part of the disk
    ax_pl.imshow(flux_zoom_pl,origin='lower',cmap='nipy_spectral',extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]],alpha=1.) 
      
    #plot disk in front of continuum
    ax_pl.imshow(disk_front_pl,origin='lower',cmap='nipy_spectral',extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]],alpha=0.1)   
    #plot disk in behind continuum
    ax_pl.imshow(disk_behind_pl,origin='lower',cmap='nipy_spectral',extent=[x_los[0],x_los[-1],y_los[0],y_los[-1]],alpha=0.4)  

    
    #set ticks
    ax_pl.set_xlabel(r'z [kpc]')
    ax_pl.set_ylabel(r'y [kpc]')
    ax_pl.set(adjustable='box-forced', aspect='equal')
    #ax_pl.set_yticks([])
    #ax_pl.set_xticklabels([-4,-3,-2,-1,0,1,2,3])
    #ax_pl.set_yticklabels([-4,-3,-2,-1,0,1,2,3])
    #set title
    ax_pl.set_title('View from `the side`',fontsize=22)     
   

    #-------------------------------------------------#
    # Save figure                                     #
    #-------------------------------------------------#   
    
    fig_a.savefig(outfile_fig,format='png',bbox_inches='tight')
    
    return 0


# ** LINE CALCULATOR: from the continuum defines the absorbed cube line and plots **

# In[321]:

def disk_one(self,continuum_cube_z,continuum_cube_z2,continuum_image,z_los,y_los,x_los):
    #-------------------------------------------------#
    #Set the input coordinates cube                   #
    #-------------------------------------------------#

    print '...set disk...\n'

    #create a meshgrid for the cube
    z_cube,y_cube,x_cube = np.meshgrid(z_los,y_los,x_los)


    #-------------------------------------------------#
    #Compute absorption                              #
    #-------------------------------------------------#

    print '...compute absorption for disk 1...\n'

    #compute the spectrum and the cubes
    spec_integral,velocity,disk_front,disk_behind=mod_abs(self,z_cube,y_cube,x_cube,continuum_cube_z,int(1))

    #-------------------------------------------------#
    #2 DISK CASE                                     #
    #-------------------------------------------------#

    if self.RMAXD2 != 0.0 :
        print '...compute absorption for disk 2...\n'

        #compute the second set of lines and cubes and 
        #merge them with the first set
        spec_int_2,velocity_2,disk_front_2,disk_behind_2=mod_abs(self,z_cube,y_cube,x_cube,continuum_cube_z2,int(2))
    
        #merge absorbed cubes
        velocity=np.nan_to_num(velocity)
        velocity_2=np.nan_to_num(velocity_2)
        velocity=np.add(velocity,velocity_2) 
        velocity[velocity==0.0]=np.nan

        #merge disks in front
        disk_front=np.nan_to_num(disk_front)
        disk_front_2=np.nan_to_num(disk_front_2)
        disk_front=np.add(disk_front,disk_front_2)    
        disk_front[disk_front==0.0]=np.nan
        
        #merge disks behind
        disk_behind=np.nan_to_num(disk_behind)
        disk_behind_2=np.nan_to_num(disk_behind_2)
        disk_behind=np.add(disk_behind,disk_behind_2)    
        disk_behind[disk_behind==0.0]=np.nan

        #merge spectrum
        spec_integral[:]+=spec_int_2[:]

    #set final integrated spectrum
    spec_int[1,:]=spec_integral[:]   
    
    #-------------------------------------------------#
    #CONVOLVE the SPECTRUM                            #
    #-------------------------------------------------#

    print '...convolve spectrum...\n'

    #convolve the spectrum at the desired velocity resolution
    spec_int[1,:]=convoluzion(self,spec_int[1,:])

    #-------------------------------------------------#
    #NORMALIZE the SPECTRUM to the OBSERVED one       #
    #-------------------------------------------------#
    
    print '...normalize spectrum...\n'

    #load observed spectrum
    spec_obs=np.loadtxt(filespec)
    #convert in frequency given the systemic velocity
    spec_obs[:,0]=(HI_hz/spec_obs[:,0]-1)*C
    spec_obs[:,0]=spec_obs[:,0]-self.VSYS
        
    #normalize modelled spectrum to observed one
    peak_obs=np.min(spec_obs[:,1])
    peak_mod=np.min(spec_int[1,:])

    spec_int_norm=np.divide(spec_int[1,:],peak_mod)
    spec_int_mod=np.multiply(spec_int_norm,peak_obs)

    #-------------------------------------------------#
    #Determine residuals and chi2                     #
    #-------------------------------------------------#

    print '...compute some stats...\n'

    spec_full=np.array([vels,spec_int_mod])

    residuals,obs_res,mod_res=chi_res(self,spec_full,spec_obs)

    # some STATISTICS on the residuals

    
    self.NOISE_RES=np.std(residuals[1,:])
    self.MED_RES=np.median(residuals[1,:])
            
    #-------------------------------------------------#
    #Write new line in output table                   #
    #-------------------------------------------------#
    print '...write table some stats...\n'

    writeTable(self,out_table)
    
     #-------------------------------------------------#
    # PLOT                                             #
    #--------------------------------------------------#

    #define smart names for the output figure
    outfile_fig=root_out+'spec_fit_'+str(int(self.I))+'_'+str(int(self.PA))+'_'+str(int(self.PA_C))+'.png'


    print '...Begin Plotting...\n'

    plot_figure(self,spec_obs,spec_int_mod,residuals,mod_res,obs_res,outfile_fig,velocity,
                continuum_image,disk_front,disk_behind)

    print '...End Plotting...\n'   
    
    
    return 0
  
###################################################
        


# ** Define the cube & the variable parameters **

# In[322]:



#-------------------------------------------------#
#CUBE                                             #
#-------------------------------------------------#

print '********************\n'   
print 'INPUT CUBE\n'

#build the cube which contains my disk based on its (RMAX)
#and on the resolution of the 1st cycle

#set the edges to RMAX+20%
x_los=np.arange(-self.RMAX*1.2,+self.RMAX*1.2+self.RES,self.RES)
y_los=np.arange(-self.RMAX*1.2,+self.RMAX*1.2+self.RES,self.RES)
z_los=np.arange(-self.RMAX*1.2,+self.RMAX*1.2+self.RES,self.RES)

#edges of the cube
print 'edges of the cube [pc]:\t\t\t'      +str(x_los[-1]),str(x_los[0])
print 'size of the cube [pixels]]:\t\t'  +str(len(x_los))+' x '+str(len(y_los))+' x '+str(len(z_los))
print 'resolution first cycle [pc]:\t\t' +str(self.RES)+'\n'

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

vels=np.arange(-self.VROT*2.,self.VROT*2.+self.VRES,self.VRES)-self.VRES/2.
spec_int=np.zeros([2,len(vels)])
spec_int[0,:]=vels[:]

#edges of the velocity array
print 'edges of the velocity array [km/s]:\t' + str(vels[-1]),str(vels[0])
print 'velocity resolution [km/s]:\t\t' + str(self.VRES)
print 'convolved velocity resolution [km/s]:\t' + str(2.*self.DISP)+'\n'
print '********************\n'   

#-------------------------------------------------#
#VARIABLE PARAMETERS                              #
#-------------------------------------------------#

#PA_vec=np.arange(self.PA_LIM[0],self.PA_LIM[-1]+self.STEP,self.STEP)
I_vec=np.arange(self.I_LIM[0],self.I_LIM[-1]+self.STEP,self.STEP)
#PA_C_vec=np.arange(self.PA_C_LIM[0],self.PA_C_LIM[-1]+self.STEP,self.STEP)
PA_C_vec=np.linspace(self.PA_C_LIM[0],self.PA_C_LIM[-1],len(I_vec))
PA_vec=np.linspace(self.PA_LIM[0],self.PA_LIM[-1],len(I_vec))

s=np.array([I_vec,PA_vec,PA_C_vec])

#grids_x,grids_y,grids_z = np.meshgrid(PA_vec,I_vec,PA_C_vec)

#print grids_x[0,-1,0],grids_y[0,0,0],grids_z[0,0,0]
#print grids_x[0],grids_z[0]
#print grids_x[1],grids_z[1]
#print grids_x[2],grids_z[2]

#print grids_x.shape,grids_y.shape,grids_z.shape

var_comb=list(itertools.product(*s))

s=np.array([I_vec,PA_vec,PA_C_vec])
grids_x,grids_y,grids_z = np.meshgrid(I_vec,PA_vec,PA_C_vec)

print 'These are the variables of the parameter space (I,PA,PA_C):'
print s



print 'READY for '+str(len(var_comb))+' runs'


###################################################



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

# In[323]:

#-------------------------------------------------#
#MAIN MAIN MAIN                                   #
#-------------------------------------------------#

print '********************\n'   
print '...Begin...\n'

#start timer
tempoinit=time.time()

#-------------------------------------------------#
#Set the input continuum cube                     # 
#-------------------------------------------------#

print '... set continuum input cube...\n'

#load the continuum image interpolated at the specified resolution
continuum_image=build_continuum(self,y_los,x_los)

#to use for uniform absorption 
#continuum_image=np.zeros([len(x_los),len(y_los)])+1.

#create a cube containing the fluxes of the continuum image

#-------------------------------------------------#
# Based on the coordinate system of the README    # 
# the axis are sorted in the array: [y,z,x]       #
#-------------------------------------------------#

continuum_cube_z=np.dstack([continuum_image]*(len(z_los)))
continuum_cube_z=np.swapaxes(continuum_cube_z,1,2)

#mask the noise of the continuum in the cube
index_mask = continuum_cube_z < self.CONT_LIM
continuum_cube_z[index_mask] = 0.0
continuum_cube_z2 = continuum_cube_z.copy()

#mask the noise of the continuum in the continuum image
index_mask = continuum_image < self.CONT_LIM        
continuum_image[index_mask] = 0.0

#-------------------------------------------------#
#ITERATION OVER PA AND I                          #
#-------------------------------------------------#


#for i in xrange(0,len(var_comb)):
    
    
    #set variables
#    self.RUN=i+1
    
#    print '********************'   

#    print "\tRUN %f" % (self.RUN)  

#    print '********************'     
    
#    self.I=var_comb[i][0]
#    self.PA=var_comb[i][1]
#    self.PA_C=var_comb[i][2]
    
    #copy continuum cube
#    cont1=continuum_cube_z.copy()
#    cont2=continuum_cube_z2.copy()
    
#    c_im=continuum_image.copy()

#    disk_one(self,cont1,cont2,c_im,z_los,y_los,x_los)
    
#    if self.RUN == 1:
#        validation_array=np.array([self.RUN,self.I,self.PA,self.PA_C,self.NOISE_RES])
#    else:
#        validation_array=np.vstack((validation_array,[self.RUN,self.I,self.PA,self.PA_C,self.NOISE_RES]))



grid_f=np.copy(grids_x)*np.nan


count=1
for i in xrange (grids_x.shape[0]):
    for j in xrange (grids_x.shape[1]):
        for k in xrange (grids_x.shape[2]):
            
                #set variables
                self.RUN=count
    
                print '********************'   

                print "\tRUN %f" % (self.RUN)  

                print '********************'     
    
                self.I=grids_x[i,j,k]
                self.PA=grids_y[i,j,k]
                self.PA_C=grids_z[i,j,k]
    
                #copy continuum cube
                cont1=continuum_cube_z.copy()
                cont2=continuum_cube_z2.copy()
    
                c_im=continuum_image.copy()

                disk_one(self,cont1,cont2,c_im,z_los,y_los,x_los)
    
                #if self.RUN == 1:
                #    validation_array=np.array([self.RUN,self.I,self.PA,self.PA_C,self.NOISE_RES])
                #else:
                #    validation_array=np.vstack((validation_array,[self.RUN,self.I,self.PA,self.PA_C,self.NOISE_RES]))
                
                count+=1
                
                grid_f[i,j,k]=self.NOISE_RES
        
        
        
        
print '********************'   

print '...End... \n'

tempofin=(time.time()-tempoinit)/60.
tempoinit=tempofin

print '********************'   

print "\tTotal time: %f minutes" % (tempofin)  

print '********************'   
print 'NORMAL TERMINATION'
print '********************\n' 

  
###################################################
# END                                             #
###################################################


# In[364]:

#-------------------------------------------------#
#PLOT for STATISTICAL ANALYSIS                    #
#-------------------------------------------------#


def plot_anal(self,grid_f,grid_x,grid_y,grid_z,s):

    
    
    
    
    #define figure parameters  
    params = {'legend.fontsize': 14,
           'axes.linewidth':2,
            'axes.labelsize':22,
           'lines.linewidth':1,
           'legend.linewidth': 3,
           'xtick.labelsize':22,
           'ytick.labelsize':22,
           'xlabel.fontsize':22,
           'ylabel.fontsize':22,
           'text.usetex': True,
           'text.latex.unicode' : True }
    rc('font',**{'family':'serif','serif':['serif']})
    plt.rcParams.update(params)
          

    #-------------------------------------------------#
    # Set FIGURE and GRID                             #
    #-------------------------------------------------#
    
    
    fig_a, ax = plt.subplots(2, 2,figsize=(11, 11))


    
    #-------------------------------------------------#
    # Plot the spectrum and parameters of the disk    #
    #-------------------------------------------------#
    
    #define subplots
    ax_pai=ax[0,0]
    ax_ic=ax[1,0]
    ax_pac=ax[1,1]
    ax_blank=ax[0,1]

    ax_blank.axis('off')

 
    #define colormap
    cm = plt.cm.get_cmap('YlOrBr_r')

    # find best FIT
    mini=np.min(grid_f)
    
    index_min= grid_f == mini    
    
    print grid_x[index_min], grid_y[index_min], grid_z[index_min]
    
    #-------------------------------------------------#
    # PLOT 1: I & PA figure                           #
    #-------------------------------------------------# 
  
    #Flattening for plot 1

    ipa=np.nanmean(grid_f,axis=2)
    
    print len(ipa)
    sort_grid=ipa.flatten()
    np.sort(sort_grid)
    
    cont10=sort_grid[int(len(sort_grid)/100.*10.)]
    
    print len(sort_grid)
    print cont10
    
    
    
    #Plot I vs PA
    ax_pai.imshow(ipa,origin='lower',cmap=cm,interpolation='lanczos',
                  extent=[s[0][0],s[0][-1],s[1][0],s[1][-1]],alpha=1.,   
                  norm=LogNorm(vmin=ipa.min(), vmax=ipa.max()))      
    ax_pai.scatter(grid_x[index_min], grid_y[index_min], marker='+',s=8e2,c='red')
    
    cont=[cont10]
    ax_pai.contour(ipa,cont,origin='lower',
                     colors='black',linewidths=3,ls='-.',extent=[s[0][0],s[0][-1],s[1][0],s[1][-1]])
    # set labels and ticks
    ax_pai.set_xlabel(r'I [$^\circ$]')
    ax_pai.set_ylabel(r'PA [$^\circ$]')
    ax_pai.xaxis.set_label_position('top') 


    # Set the ticks and labels...
    #ticks_i = np.linspace(x_val.min(), x_val.max(), 6)
    #ax_pai.set_xticklabels(ticks_i)
    
    #ticks_pa = np.linspace(y_val.min(), y_val.max(), 6)
    #ax_pai.set_yticklabels(ticks_pa)

    
    ax_pai.xaxis.tick_top()
    ax_pai.set_aspect('auto')

    ax_pai.set(adjustable='box-forced')
    
    #-------------------------------------------------#
    # PLOT 2: I & PA_C figure                         #
    #-------------------------------------------------# 
   
    #Flattening for plot 2

    ic=np.nanmean(grid_f,axis=1)
    
    ic=numpy.flipud(ic)

    sort_grid=ic.flatten()
    np.sort(sort_grid)    
    cont10=sort_grid[int(len(sort_grid)/100.*10.)]
    
    
    #ic=np.swapaxes(ic,0,1)
    #Plot I vs PA continuum
    ax_ic.imshow(ic,origin='lower',cmap=cm,interpolation='lanczos',
                  extent=[s[0][0],s[0][-1],s[2][0],s[2][-1]],alpha=1., vmin=ic.min(),vmax=ic.max(),  
                  norm=LogNorm(vmin=ic.min(), vmax=ic.max()) )      
    ax_ic.scatter(grid_x[index_min], grid_z[index_min], marker='+',s=8e2,c='red')

    ax_ic.contour(ic,cont,origin='lower',
                     colors='black',linewidths=3,ls='-.',extent=[s[0][0],s[0][-1],s[2][0],s[2][-1]])
    
    #set ticks and labels
    ax_ic.set_xlabel(r'I [$^\circ$]')
    ax_ic.set_ylabel(r'PA continuum [$^\circ$]')

    ax_ic.set_aspect('auto')
    ax_ic.set(adjustable='box-forced')

    #-------------------------------------------------#
    # PLOT 3: PA & PA_C figure                        #
    #-------------------------------------------------# 

    #Flattening for plot 3
    
    pac=np.nanmean(grid_f,axis=0)

    pac=numpy.flipud(pac)

    
    sort_grid=pac.flatten()
    np.sort(sort_grid)    
    cont10=sort_grid[int(len(sort_grid)/100.*10.)]
    
    print int(len(sort_grid)/100.*10.)
    
    #plot PA vs PA continuum
    ax_pac.imshow(pac,origin='lower',cmap=cm,interpolation='lanczos',
                  extent=[s[1][0],s[1][-1],s[2][0],s[2][-1]],alpha=1.,
                  norm=LogNorm(vmin=pac.min(), vmax=pac.max()))       
    ax_pac.scatter(grid_y[index_min], grid_z[index_min], marker='+',s=8e2,c='red')
   
    ax_pac.contour(pac,cont,origin='lower',
                     colors='black',linewidths=3,ls='-.',extent=[s[1][0],s[1][-1],s[2][0],s[2][-1]])
    
    #set ticks and labels
    ax_pac.set_xlabel(r'PA [$^\circ$]')
    ax_pac.set_ylabel(r'PA continuum [$^\circ$]')
    ax_pac.yaxis.tick_right()
    ax_pac.yaxis.set_label_position('right') 

    ax_pac.set_aspect('auto')
    ax_pac.set(adjustable='box-forced')
    
    fig_a.subplots_adjust(wspace=0, hspace=0)
    
    fig_a.show()
    #-------------------------------------------------#
    # Save figure                                     #
    #-------------------------------------------------#   
    outfile_fig=root_out+'val_par.png'
    
    fig_a.savefig(outfile_fig,format='png',bbox_inches='tight')  
    

    return 0


# In[365]:

#-------------------------------------------------#
# READ table and plot validation plots            #
#-------------------------------------------------#



#LOAD from TABLE
#val_arr = np.genfromtxt(out_table, delimiter=',', names=True,dtype=None)
#validation_array=np.array([val_arr['Idegrees'],val_arr['PAdegrees'],
#                           val_arr['PA_contdegrees'],val_arr['disp_res']])
#f_val=validation_array[3,:]
#a=grids_x.shape
#f_val=f_val.reshape(a)

#grid_f=np.copy(grids_x)*np.nan

#for i in xrange (grids_x.shape[0]):
#    for j in xrange (grids_x.shape[1]):
#        for k in xrange (grids_x.shape[2]):

#            grid_f[i,j,k]=f_val[i,j,k]



print '********************'   
print s
print '********************'   



plot_anal(self,grid_f,grids_x,grids_y,grids_z,s)

print 'NORMAL TERMINATION'


# In[ ]:




# In[ ]:



