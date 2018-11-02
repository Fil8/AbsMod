import emcee
import numpy as np
import random
from scipy.interpolate import interp1d


import disk as disk
import stats as stats
import spectrum as spec



# ----------------#
def chi_square(s_mod, s_obs, cfg_par):
    # interpolate to have arrays all of the same length
    func_obs = interp1d(s_obs[:, 0], s_obs[:, 1])
    func_mod = interp1d(s_mod[0, :], s_mod[1, :])

    VROT = cfg_par['vel_pars'].get('vrot', 200)
    DISP = cfg_par['vel_pars'].get('disp', 200)

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


def lnprior(theta, cfg_par):
    inc, pos_ang = theta

    key = 'mcmc_pars'
    I_d = cfg_par[key].get('I_left', 0.)
    I_u = cfg_par[key].get('I_right', 90.)

    PA_d = cfg_par[key].get('PA_left', 0.)
    PA_u = cfg_par[key].get('PA_right', 180.)

    if I_d < inc < I_u and PA_d < pos_ang < PA_u:
        return 0.0
    return -np.inf


# ----------------#


def lnprob(theta, continuum_cube_z, spec_obs, vels, spec_int, cfg_par):
    lp = lnprior(theta, cfg_par)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, continuum_cube_z, spec_obs, vels, spec_int, cfg_par)


# ----------------#


def lnlike(theta, continuum_cube_z_ln, spec_obs_ln, vels, spec_int, cfg_par):
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

    # open output table
    cfg_par['disk_1']['i'] = inc 
    cfg_par['disk_1']['pa'] = pos_ang 

    # -------------------------------------------------#
    # Compute absorption                               #
    # -------------------------------------------------#

    # compute the spectrum and the cubes
    spec_integral, velocity, disk_front, disk_behind  = disk.mod_abs(continuum_cube_z_ln, vels, int(1), cfg_par)

    # set final integrated spectrum
    spec_int[1, :] = spec_integral[:]

    # -------------------------------------------------#
    # CONVOLVE the SPECTRUM                            #
    # -------------------------------------------------#

    #print '...convolve spectrum...\n'

    DISP = cfg_par['vel_pars'].get('disp', 200)

    spec_int = stats.convoluzion(spec_int,DISP)
    spec_int = np.array([vels, spec_int])
    #print cfg_par['disk_1']
    # convolve the spectrum at the desired velocity resolution
    # spec_int[1,:]=smoothing_func(spec_int[1,:])

    # -------------------------------------------------#
    # NORMALIZE the SPECTRUM to the OBSERVED one       #
    # -------------------------------------------------#
    if cfg_par['mcmc_pars']['flat'] == False:
    
        spec_int_mod = stats.normalize(spec_int,spec_obs_ln)
    else:
        spec_int_mod = spec_int[1,:].copy()
    # -------------------------------------------------#
    # Determine residuals & chi2 & widths of the line  #
    # -------------------------------------------------#

    spec_full = np.array([vels, spec_int_mod])

    FWHM, multipeak,FW20 = stats.widths(cfg_par,spec_full,inc,pos_ang)
    line_pars = {'FWHM':np.round(FWHM,2), 'multipeak': np.round(multipeak,2),'FW20': np.round(FW20,2)}
    cfg_par['line_pars'] = line_pars
    print cfg_par['line_pars']

    res, mod_res, obs_res = chi_square(spec_full, spec_obs_ln, cfg_par)

    noise = np.std(obs_res[1, :])
    inv_noise = 1. / (noise * noise)

    if cfg_par['mcmc_pars']['flat'] == False:
        loglike = -0.5 * np.sum((np.power(res[1, :], 2) * inv_noise - np.log(inv_noise)))
    else:
        loglike = 1.0
    


    # -------------------------------------------------#
    #write table
    RUN = 0

    return loglike

#--------------------------------------------------#
#  MCMC                                            #
#--------------------------------------------------#
def run_sim(continuum_cube_z, cfg_par):

    vels, spec_int, spec_obs = spec.input_spectrum(cfg_par)


    key = 'mcmc_pars'
    I_d = cfg_par[key].get('I_left', 0.)
    I_u = cfg_par[key].get('I_right', 90.)

    PA_d = cfg_par[key].get('PA_left', 0.)
    PA_u = cfg_par[key].get('PA_right', 180.)

    DIM_mcmc = cfg_par[key].get('ndim_mcmc', 2)
    WALK_mcmc = cfg_par[key].get('nwalkers_mcmc', 40)
    STEPS_MCMC = cfg_par[key].get('nsteps_mcmc', 100)
    #PA_d += 90.
    #PA_u += 90.

    p0 = []
    for i in xrange(WALK_mcmc):
        inclinations = random.uniform(I_d, I_u)
        positionangles = random.uniform(PA_d, PA_u)

        p0.append((inclinations, positionangles))

    print '... starting seeds ...'
    print p0

    # run mcmc algorithm
    sampler = emcee.EnsembleSampler(WALK_mcmc, DIM_mcmc, lnprob,
                                    args=(continuum_cube_z, spec_obs, vels, spec_int, cfg_par),
                                    threads=15)
    
    print '... start walking ...'

    sampler.run_mcmc(p0, STEPS_MCMC)

    print '... save samples & walkers ...'
    #load data
    
    #-------------------------------------------------#
    #Write new line in output table                   #
    #-------------------------------------------------#
    GAL = cfg_par['general'].get('model_name', '1')
    workdir = cfg_par['general'].get('work_directory', None)
    root_out = workdir+'outputs/'

    print '...write table with stats...\n'
    out_table = root_out+'table_out_mcmc.csv'
    RUN = stats.write_table_mcmc(out_table, cfg_par)

    samples_text = root_out+GAL+'_'+str(RUN)+'_samples.txt'
    walkers_array = root_out+GAL+'_'+str(RUN)+'_walkers.npy'
    samples = sampler.chain[:, :].reshape((-1, DIM_mcmc))
    #samples[:, 1]-= 90.
    np.savetxt(samples_text, samples)
    samp_chain = np.array(sampler.chain[:, :, :], dtype=float)
    #samp_chain[:, :, 1]-=90.
    np.save(walkers_array, samp_chain)

    return RUN



