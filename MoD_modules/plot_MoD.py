import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import corner

import disk as disk

#-------------------------------------------------#
# PLOT function                                   #
#-------------------------------------------------#


def plot_figure(spec_obs, spec_plt, res, outfile_fig,
                flusso, continuum_image, disk_front, disk_behind,cfg_par):

    plt.ioff()

    RMAX = cfg_par['disk_1'].get('rmax', 1000)
    VROT = cfg_par['vel_pars'].get('vrot', 200)
    CONT_LIM = cfg_par['abs_pars'].get('flux_cont_lim', 1e-3)
 
    x_los, y_los, z_los = disk.main_box(cfg_par)


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

def walkers_plot(walkers, out_fig_walkers_name, cfg_par):

    key = 'mcmc_pars'
    DIM_mcmc = cfg_par[key].get('ndim_mcmc', 2)
    WALK_mcmc = cfg_par[key].get('nwalkers_mcmc', 40)
    STEPS_MCMC = cfg_par[key].get('nsteps_mcmc', 100)
    I_d = cfg_par[key].get('I_left', 0.)
    I_u = cfg_par[key].get('I_right', 90.)

    PA_d = cfg_par[key].get('PA_left', 0.)
    PA_u = cfg_par[key].get('PA_right', 180.)   
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

    #i_tick_array = [0, 15, 30, 45, 60, 75, 90]
    #i_tick_labels_array = [str(0), str(15), str(30), str(45), str(60), str(75), str(90)]
    
    #i_tick_array = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    #i_tick_labels_array = [str(0), str(20), str(40), str(60), str(80), str(100), str(120), 
    #                       str(140), str(160), str(180)]
    
    i_label_name = r'I [$^\circ$]'
    
    #pa_tick_array = [0, 30, 60, 90, 120, 150, 180]
    #pa_tick_labels_array = [str(0), str(30), str(60), str(90), str(120), str(150), str(180)]
    
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
    
    #wal_i.set_yticks(i_tick_array)
    #wal_i.set_yticklabels(i_tick_labels_array)    
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
    
    #wal_pa.set_yticks(pa_tick_array)
    #wal_pa.set_yticklabels(pa_tick_labels_array)    
    wal_pa.set_ylabel(pa_label_name)

    #-------------------------------------------------#
    # Save figure                                     #
    #-------------------------------------------------#   
    
    fig_a.savefig(out_fig_walkers_name, format='png', bbox_inches='tight')  

    return 0


def corner_plot(samples, out_fig_samples_name, cfg_par):

    key = 'mcmc_pars'
    DIM_mcmc = cfg_par[key].get('ndim_mcmc', 2)
    WALK_mcmc = cfg_par[key].get('nwalkers_mcmc', 40)
    STEPS_MCMC = cfg_par[key].get('nsteps_mcmc', 100)
    I_d = cfg_par[key].get('I_left', 0.)
    I_u = cfg_par[key].get('I_right', 90.)

    PA_d = cfg_par[key].get('PA_left', 0.)
    PA_u = cfg_par[key].get('PA_right', 180.)
    x = samples[:, 0]
    y= samples[:, 1]

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
        print "Too few points to create valid contours"
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
    
    #x_tick_array = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    #x_tick_labels_array = [str(0), str(20), str(40), str(60), str(80), str(100), str(120), 
    #                       str(140), str(160), str(180)]
    x_label_name = r'I [$^\circ$]'
    
    #y_tick_array = [0, 30, 60, 90, 120, 150, 180]
    #y_tick_labels_array = [str(0), str(30), str(60), str(90), str(120), str(150), str(180)]
    
    #y_tick_array = [0, 60, 120, 180, 240, 300, 360]
    #y_tick_labels_array = [str(0), str(60), str(120), str(180), str(240), str(300), str(360)]
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
    #ax_cont.set_xticks(x_tick_array)
    #ax_cont.set_xticklabels(x_tick_labels_array)    
    ax_cont.set_xlabel(x_label_name)
    
    #ax_cont.set_yticks(y_tick_array)
    #ax_cont.set_yticklabels(y_tick_labels_array)    
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
    #ax_i.set_xticks(x_tick_array)
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
    #ax_pa.set_xticks(y_tick_array)
    #ax_pa.set_xticklabels(y_tick_labels_array)    
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
    
    fig_a.savefig(out_fig_samples_name, format='png', bbox_inches='tight')  

    return 0