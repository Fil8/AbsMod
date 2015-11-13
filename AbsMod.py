
# coding: utf-8

#!/usr/bin/env python
import math
import time
import numpy as np
import string,sys,os
import argparse
from matplotlib import pyplot as plt
import pyfits
from astropy import wcs
from astropy.io import fits



# #UPLOAD PARAMETERS of the Disk:
# 
# - read the parameter file, located in rootdir.
# - create the dictionary for the parameters.
# - write a file .fits with all the features of the cube in the header (wcs coords etc etc....)

# some important functions:

def readFile(parameterFile):
        
    parameters={}
    
    try:
        paramFile=open(parameterFile)
    except:
        print "%s file not found"  % parameterFile
        return 1
        
    paramList=paramFile.readlines()
    paramFile.close()     

    
    for line in paramList:
        if line.strip():  # non-empty line?
            tmp=line.split('=')
            tmp2=tmp[0].split('[')
            key, value = tmp2[0],tmp[-1]  # None means 'all whitespace', the default
            parameters[key] = value
    
    
    return parameters

def ra2deg(rad):
    
        ra=string.split(rad,':')
    
        hh=float(ra[0])*15
        mm=(float(ra[1])/60)*15
        ss=(float(ra[2])/3600)*15
        
        return hh+mm+ss

def dec2deg(decd):
        dec=string.split(decd,':')
        
        hh=abs(float(dec[0]))
        mm=float(dec[1])/60
        ss=float(dec[2])/3600
        return hh+mm+ss

def flux_read(x_los,y_los):
        
        x_los/=scale_cont_pc
        y_los/=scale_cont_pc
        y1=cen_y+y_los
        x1=cen_x+x_los
        
        #continuum value linear interpolation
        
        a=math.modf(x1)
        b=math.modf(y1)
        x1=int(a[1])
        x2=int(a[1])+1
        xw1=1-a[0]
        xw2=a[0]
        y1=int(b[1])
        y2=int(b[1])+1
        yw1=1-b[0]
        yw2=b[0]
    
        cont_value=(dati[y1,x1]*xw1*yw1+dati[y1,x2]*xw2*yw1+dati[y2,x1]*xw1*yw2+dati[y2,x2]*xw2*yw2)/(xw1*yw1+xw1*yw2+xw2*yw2+xw2*yw1) 
        
        return cont_value


def ang2lin(z,dl,ang): # r in arcsec
    
    #dl = dl/3.085678e24 # Mpc
    r = ang * dl / (RAD2DEG * 3600 * (1+z)**2) # Mpc
    
    return r

def lin2ang(z,dl,r): # r in Mpc

    ang = RAD2DEG * 3600. * r * (1.+z)**2 / dl # arcsec

    return ang

#Functions for the disk

def flux(r):

    rho=FLUX_VALUE
    flux=+rho
    
    return flux    

def space(x_los,y_los,z_los):

        #allocate the output array
        v_z=np.array([0.0,0.0])

        #transformation into disk coordinates
        x=trig_PA[1]*x_los+trig_PA[0]*y_los
        y=trig_I[1]*(-trig_PA[0]*x_los+trig_PA[1]*y_los)+trig_I[0]*z_los
        z=-trig_I[0]*(-trig_PA[0]*x_los+trig_PA[1]*y_los)+trig_I[1]*z_los
        #determine the radius of the disk
        r=np.sqrt(x*x+y*y)

        #rotation curve rising within RMIN
        #RMIN=0.5
        #m=VROT/RMIN
        #if r <= RMIN:
        #    velo=r*m
        #else:
        velo=VROT        
        
        #condition I am in the disk
        if r <=RMAX and abs(z)<=H_0/2. and r>=RMIN:
            angle=math.atan2(y,x)
            v_z[0]=-SIGN*trig_I[0]*np.cos(angle)*velo
            v_z[1]=flux(r)
        else:
            #If I am outside the disk I set a velocity outside of its range (to improve…)
            v_z[0]=1e4

        return v_z  

def cut_through_pl(xx_los,yy_los,zstop):
    spec_pl=np.zeros(len(vels))
    

    z_los=z_start
    zstep=RES
    i=0
    while (z_los<=zstop):
            v1=space(xx_los,yy_los,z_los)
            if v1[0] != 1e4:
                
                ind = (np.abs(vels - (v1[0]))).argmin()
                spec_pl[ind]=spec_pl[ind]+v1[1]
            z_los=z_los+zstep
            i+=1
    #print spec_pl
    if np.sum(spec_pl)!=0.0:
        massimo_ind=np.argmax(spec_pl)
        massimo=vels[massimo_ind]
        massimo_norm=(massimo-(-VROT))/(VROT*2.)
        #print massimo
        #sprint massimo_norm

    else:
        massimo_norm=1e4
    print i
    return massimo_norm
    
  
def cut_through(xx_los,yy_los,flux_cont,zstop):
    
    cond=1e6
    spec_wide=np.zeros(len(vels))
    spec_thin=np.zeros(len(vels))
    zstep=RES
    #initial condition on the cut through 
    z_los=z_start-RES
    niters=0
    vektor=[]
    #check spectra
    while ( (cond>CONDITION) and (niters<20)):
    #cut through to make spectra
        while (z_los<=zstop):
            z_los=z_los+zstep
            v1=space(xx_los,yy_los,z_los)
            if v1[0] != 1e4:
                
                ind = (np.abs(vels - (v1[0]))).argmin()
                spec_thin[ind]=spec_thin[ind]+v1[1]
    
    
        spec_thin=np.convolve(spec_thin,gauss,mode='same')
    
        #check spectra and determine condition for good resolution
        condition_under=0.0
        for i in xrange(0,len(vels)):
            spec_thin[i]=spec_thin[i]/(2.0**niters)
            condition_under = max(condition_under,math.fabs(spec_thin[i]-spec_wide[i]))
                        
        cond=condition_under*zstep
        #set conditions for next loop at higher resolution
        spec_wide=spec_thin.copy()    
    
        zstep=zstep/2.
        z_los=z_start-zstep
    
        #next loop
        niters+=1
    
    niters=niters-1
    a_line= np.sum(spec_thin)
    a_tot=flux_cont-a_line
    a_tot=flux_cont-a_tot
    #print a_line
    
    if a_line != 0.0:
        spec_thin[:]=-spec_thin[:]/a_line*a_tot
    else:
        pass
    return spec_thin,niters,a_line  
    
    
    
def plotta(xdim,ydim):
    
    #fig = plt.figure(figsize=(15,7))
   
            

    #a.plot(specint[0,:],specint[1,:],color='r')

    #plt.show() 
    

 
    #cm_back = plt.cm.get_cmap('gray')

    #cm = plt.cm.get_cmap('gray')

    #ax1=plt.subplot2grid((1,3),(0,0))
    #ax2=plt.subplot2grid((1,3),(0,1))
    #ax3=plt.subplot2grid((1,3),(0,2))
    y_los=ystart
    
    RES_PLT=RES*3.
    x=[]
    y=[]
    col=[]
    mas=np.zeros([ydim/3+1,xdim/3+1])
    cubim=np.nan*np.zeros([ydim/3+1,xdim/3+1])
    print xdim,ydim
    y_los_coord=0
    while (y_los<ystop):
        #disk... from left to right 
        x_los=xstart
        x_los_coord=0

        while (x_los<xstop):  
            
            zstop=cont_tan *(x_los)
            print zstop
            x_los_cont=x_los/scale_cont_pc
            y_los_cont=y_los/scale_cont_pc
            y1=cen_y+y_los_cont
            x1=cen_x+x_los_cont
            massimo_norm=cut_through_pl(x_los,y_los,zstop)

            mas[y_los_coord,x_los_coord]=mask[y1,x1]
            if massimo_norm!=1e4:
                cubim[y_los_coord,x_los_coord]=massimo_norm

            
            #if massimo_norm!=1e4:
                col.append(massimo_norm)
                x.append(x_los)
                y.append(y_los)
            
            x_los+=RES_PLT
            x_los_coord+=1
        y_los+=RES_PLT
        y_los_coord+=1
    #print mas
    #print np.sum(mas)
    fig = plt.figure(figsize=(14,9))
    tick_size=16
    lines=12
    #a = fig.add_subplot(111,xlabel='V',ylabel='S',                    
    #                autoscalex_on=True,autoscaley_on=True)        
    params={'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size,'axes.linewidth' : 3,'legend.labelspacing':1,'legend.linewidth':3,'legend.fontsize':14}
 
    plt.rcParams.update(params)

        #normalize for colorscale
        
    cm_vel = plt.cm.get_cmap('gray')    
    #ax = axes([0,0,1,1], frameon=False)
    #ax.set_xlim(xstart,xstop)
    #ax.set_ylim(ystart,ystop)
    plt.imshow(mas[:,:], cmap='jet',origin='lower',extent=[xstart,xstop,ystart,ystop])   
    plt.imshow(cubim[:,:], cmap='jet',origin='lower',extent=[xstart,xstop,ystart,ystop])
    #plt.colorbar(asa) ## 

    #plt.scatter(x,y, marker='o',c=col,cmap=cm_vel,lw=0)
    #cbar=plt.colorbar(im) ## 

    plt.show()
    outfile=rootdir+'spedisk.png'
    plt.savefig(outfile)
    plt.close()
    
    
        
 

#####START#####
tempoinit=time.time()



##### UPLOAD ###### 
#rootdir='/data/users/maccagni/Model/'

#rootspec=rootdir+'spectra_pix/'

rootdir='/Users/maccagni/Documents/PhD/WSRT-Stacking/HI_datareduction/zoo/coralZ/Model/'
rootspec=rootdir+'spectra_pix/'


RAD2DEG=180./math.pi


#READ the parameter file
#when calling from inputfile 
#file=sys.argv[1]
fileinput = rootdir+'par.txt'

#set a dictionary with the parameters -> call readFile function
par=readFile(fileinput)
#set the variables
RMAX=float(par.get('rmax'))
RMIN=float(par.get('rmin'))
VROT=float(par.get('vrot'))
H_0=float(par.get('h0'))
CONDITION=float(par.get('condition'))
CONDITION1=float(par.get('condition2d'))
SIGN=float(par.get('sign'))
I=float(par.get('i'))
PA=float(par.get('pa'))
FLUX_VALUE=float(par.get('flux'))
D_L=float(par.get('d_l'))
z_red=float(par.get('z'))
RA=par.get('ra')
DEC=par.get('dec')
RES=float(par.get('pix_res'))
VRES=float(par.get('vel_res'))
DISP=float(par.get('disp'))
PA_CONT=float(par.get('pa_cont'))


ra=ra2deg(RA)
dec=dec2deg(DEC)

#define the trigonometric parameters of the disk
I_rad=math.radians(I)
PA_rad=math.radians(PA)

trig_I=[np.sin(I_rad),np.cos(I_rad)]
trig_PA=[np.sin(PA_rad),np.cos(PA_rad)]
cont_tan=np.tan((PA_CONT))

print 'Variables set\n'
for keys,values in par.items():
    print keys+' = '+str(values)


#set EDGES of the CUBE

#vertical
YY=RES
ystart = -RMAX-RES-RES/2.
ystop=+RMAX+RES+RES/2.
#ystep =0.5

#horizontal
XX=RES
xstart = -RMAX -RES-RES/2.
xstop = +RMAX+RES+RES/2.

#depth
z_stop=+0.0
z_start=-RMAX-RES/2.
ZZ=RES

#define the velocity array of the OUTPUT SPECTRUM
V_eR=VROT*3.
V_m=V_eR/2.
V_dim=V_eR/VRES

vels=np.linspace(0,300,300,endpoint=False)-150.5

print vels
print 'Cube set'

#DEFINE THE GAUSSIAN FOR THE CONVOLUTION
arg=-((vels*vels)/(2*DISP*DISP))
gauss=1./(np.sqrt(2*np.pi)*DISP)*np.exp(arg)

#plot disk in projection


# Here the program begins: 
# 
# - i cut through the cube fixing y coordinate and moving in the horizontal direction (x), then going to the next line. 
# - i cut through the cube (z direction) from $-2\times R_{\rm MAX}$ to $2\times R_{\rm MAX}$
# - for each line of sight I half the resolution and check if the spectra satisfy the condition
# - the condition is given by the maximum absolute difference

#####MAIN MAIN MAIN  ######


print 'Beginning'

#initial conditions on the output spectrum 
specint=np.zeros([2,len(vels)])
specint[0,:]=vels[:]
  
#define output cube 
xdim=int((xstop-xstart)/RES)
ydim=int((ystop-ystart)/RES)
cube=np.zeros([len(vels),ydim,xdim])  


scale_cube=lin2ang(z_red,D_L,RES*1e-6)/3600
print scale_cube
 
#load continuum and define the centre
file='293c.fits'
f=pyfits.open(file)
dati=f[0].data
head=f[0].header
dati=np.squeeze(dati)
dati=np.squeeze(dati)
	
	
scale_cont_asec=head['CDELT2']*3600
scale_cont_pc=ang2lin(z_red,D_L,scale_cont_asec)*1e6
print 'Continuum pixel size [pc]'
print scale_cont_pc
	
head=fits.getheader(file)
	
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

w=wcs.WCS(head)    

cen_x,cen_y=w.wcs_world2pix(ra,dec,0)

print '**** Continuum centre ****' 
print cen_x,cen_y
print '\n'
    
file='293mask.fits'
f=pyfits.open(file)
mask=f[0].data
head=f[0].header
mask=np.squeeze(mask)
mask=np.squeeze(mask)
 
plotta(xdim,ydim)
 
    
#disk... from bottom to top
y_los=ystart

y_los_coord=0
while (y_los<ystop):
    
    #disk... from left to right 
    x_los=xstart
    x_los_coord=0

    while (x_los<xstop): 
        #initial conditions on the arrays
        cond2d=1e6
        niters2d=1.
        xstep=RES
        ystep=RES
        
        #set edges of the pixel
        x_left=x_los-RES/2.
        x_right=x_los+RES/2.
        y_down=y_los-RES/2.
        y_up=y_los+RES/2.

        #extract spectrum from centre of the pixel
        
        #continuum conditions
        x_los_cont=x_los/scale_cont_pc
        y_los_cont=y_los/scale_cont_pc
        y1=cen_y+y_los_cont
        x1=cen_x+x_los_cont
        #print mask[y1,x1]
        
        flux_cont=flux_read(x_los,y_los)
        
        zstop=cont_tan *(x_los)

        spettro,nitro,ss=cut_through(x_los,y_los,flux_cont,zstop)
        
        if mask[y1,x1] ==0.0 or ss==0.0:
            
            x_los+=RES 
            continue
        else:
            
            #determine the flux of the spectrum extracted from the centre of pixel
            
            #check if i'm inside the disk with the spectrum from the centre of the pixel
            print '******NEW PIX****'
            print x_los,y_los
            print '*******'
            
            #set the spectrum from the central line of sight as the first reference spectrum
            spettro_wide=spettro.copy()

            while ((cond2d>CONDITION1) and (niters2d<5)):

                #start point along the x
                y_loss=y_down
                
                #set initial conditions: the centre of the pixel has one line of sight
                spettro_thin=spettro.copy()
                numero_media=1.

                i=0
                while (y_loss<y_up+.0001):
                    x_loss=x_left
                    while (x_loss<x_right+.0001):
                            
                                flux_cont=flux_read(x_loss,y_loss)

                                spettro_out=np.zeros(len(vels))
                                spettro_out,nitrogen,ss=cut_through(x_loss,y_loss,flux_cont,zstop)
                                #if ss!=0.0:
                                #    pass

                                #print x_loss,y_loss, numero_media
                                #print '********'

                                spettro_thin[:]+=spettro_out[:]
                            
                            
                                numero_media+=1.
                                i+=1
                                x_loss+=xstep

                            
                    y_loss+=ystep


                
                #normalize the spectrum
                spettro_thin=np.divide(spettro_thin,float(numero_media))
                condition_under2d=0.0

                for i in xrange(0,len(vels)):
                    condition_under2d = max(condition_under2d,math.fabs(spettro_thin[i]-spettro_wide[i]))
        
                cond2d=condition_under2d*xstep
            
                print cond2d,numero_media,niters2d

                #increase resolution for next step
                niters2d+=1.
                xstep=RES/niters2d
                ystep=RES/niters2d

                #update low res spectrum
                spettro_wide=spettro_thin.copy() 
            


            
            #sum into the total spectrum
            specint[1,:]+=spettro_thin[:]
            cube[:,y_los_coord,x_los_coord]=spettro_thin[:]

            print 'new spectrum'
            print numero_media
            print '\n'
            #save the pixel spectrum
            acab=np.column_stack((vels,specint[1,:]))
            outtxt=rootspec+'spec_'+str(x_los)+'_'+str(y_los)+'.txt'
            np.savetxt(outtxt,acab)
            #save the pixel spectrum
            #acab=np.column_stack((vels,spettro_thin[:]))
            #outtxt=rootspec_pixtot+'spec_'+str(x_los)+'_'+str(y_los)+'.txt'
            #np.savetxt(outtxt,acab)            
 
        #move right
        
        x_los+=RES 
        x_los_coord+=1

    #move up
    y_los+=RES
    y_los_coord+=1
    
#save the total spectrum
a=np.column_stack((vels,specint[1,:]))
outtxt=rootspec+'spectot.txt'
np.savetxt(outtxt,a)

print 'end \n'

tempofin=(time.time()-tempoinit)/60.
tempoinit=tempofin
print "total time %f minutess" % (tempofin)  
  
print 'NORMAL TERMINATION'

#prepare the outputcube

pix_x=cube.shape[2]/2
pix_y=cube.shape[1]/2

proj_cube = pyfits.PrimaryHDU()

print '####### MAKE CUBE FILES ######'
head=proj_cube.header
head.set('NAXIS',3,after='BITPIX')
head.set('BITPIX',-32,before='NAXIS')
head.set('NAXIS1',cube.shape[2],after='NAXIS')
head.set('NAXIS2',cube.shape[1],after='NAXIS1')
head.set('NAXIS3',cube.shape[0],after='NAXIS2')
head.set('BSCALE',1,after='EXTEND')
head.set('BZERO',0,after='BSCALE')
head.set('BUNIT','JY',after='BZERO')
head.set('BTYPE','intensity',after='BUNIT')
head.set('CRPIX1',pix_x,after='BTYPE')
head.set('CDELT1',-scale_cube,after='CRPIX1')
head.set('CRVAL1',ra-360.,after='CDELT1')
head.set('CTYPE1','RA--SIN',after='CRVAL1')
#head.set('CUNIT1','PC',after='CTYPE1')
head.set('CRPIX2',pix_y,after='CTYPE1')
head.set('CDELT2',scale_cube,after='CRPIX2')
head.set('CRVAL2',dec,after='CDELT2')
head.set('CTYPE2','DEC--SIN',after='CRVAL2') 
#head.set('CUNIT2','PC',after='CTYPE2') 
head.set('CRPIX3',1,after='CTYPE2')
head.set('CDELT3',VRES,after='CRPIX3')
head.set('CRVAL3',vels[0],after='CDELT3')
head.set('CTYPE3','VEL',after='CRVAL3')
head.set('CUNIT3','m/s',after='CTYPE2')
outfile=rootdir+'cubo_prova.fits'
#command='rm -r '+outfile

pyfits.writeto(outfile,cube/1e3,head.copy(),clobber=True)



fig = plt.figure(figsize=(15,7))
a = fig.add_subplot(111,xlabel='V',ylabel='S',                    
                     autoscalex_on=False, xlim=(-150,150),autoscaley_on=True)
            

a.plot(specint[0,:],specint[1,:],color='r')
outfile=rootdir+'spec_tot24.png'
plt.savefig(outfile)
#plt.show() 
