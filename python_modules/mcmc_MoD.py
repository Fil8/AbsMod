import emcee
import numpy as np

import disk as disk
import stats as stats

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
    spec_integral = disk.mod_abs(z_cube_ln, y_cube_ln, x_cube_ln, continuum_cube_z_ln, inc, pos_ang)

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

    FWHM, FW20 = stats.widths(spec_full)

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

#--------------------------------------------------#
#  MCMC                                            #
#--------------------------------------------------#
def run_sim(cfg_par):

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