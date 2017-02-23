#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
__author__ = "Laurent Perrinet INT - CNRS"
__licence__ = 'BSD licence'
DEBUG = True
DEBUG = False
"""
MotionParticlesFLE.py

Script file for MotionParticles

See http://invibe.net/LaurentPerrinet/Publications/Perrinet12pred

"""
import numpy as np
#from NeuroTools import check_dependency
import matplotlib as mpl
mpl.use("Agg") # agg-backend, so we can create figures without x-server (no PDF, just PNG etc.)
#### default parameters
# TODO: describe parameters
#from NeuroTools.parameters import ParameterSet
##from NeuroTools.parameters import ParameterRange
##from NeuroTools.parameters import ParameterTable
#p = ParameterSet({})
#import sys
#sys.settrace

import os
PID, HOST = os.getpid(), os.uname()[1]
LOCK = '_lock' + '_pid-' + str(PID) + '_host-' + HOST
import time
sleeping_time=0.2
sleeping_time=0.05
# time.sleep(np.random.rand()*sleeping_time)
recompute=False # by default, cache data in the mat folder
#recompute=True # uncomment this line to change thisf behaviour
if DEBUG: # DEBUG
    size_X, size_Y, size_T, size_N = 5, 5, 6, 6
    N_X, N_Y, N_frame = 2**size_X, 2**size_Y, 2**size_T
    N_particles = 2**size_N
else:
    size_X, size_Y, size_N = 7, 7, 12
    N_X, N_Y, N_frame = 2**size_X, 2**size_Y, 100
    N_particles = 2**size_N
latency = int(100/1000*N_frame) # 100ms in number of frames
############################ CONDENSATION parameters ############################
# diffusion coefficients
# TODO : this is a rate per frame: we should make it a rate per period
D_V = 1. # relative to one spatio temporal period  TODO : show that it is propto 1 / relaxation time
D_x = 1. # relative to one spatial period
sigma_I = .25
sigma_noise = 0.05 # std of the assumed background noise in images
p_epsilon = 0.1 # a priori probability for being on signal
sigma_motion = .1 # std of error in motion energy - intrinsic + extrinsic
K_motion = .001
# TODO : threshold for the resampling
resample = .5 # how much turn-over in the resampling method: tunes the compromise between generality and precision (it is a rate per frame)
v_init = 3. # characteristic value for the choice of v_init particles (at init + filtering)
##################################################################################
# OBSOLETE
# sigma_RF = .03
bootstrap = 1 # (int) initially, we multiply the number of particles by this number to get a better initial estimate. mainly esthetical...
polar_init = True # setting the shape of the pdf for the initial choice of particles
v_prior = 1000. # prior for slow speeds: above 10; it is as if there is no prior.
##########trials ###############
if DEBUG: # DEBUG
    N_trials = 2
else:
    N_trials = 20
########################### stimulus parameters ############################
im_noise = 0.05 # std of the background noise in images
im_contrast = 1.
width = 2. # physical horizontal width of the image (like visual angle in degrees). often 2. so that it corresponds to [-1,1]. the vertical size is width * np.float(n_y) / n_x
if not N_X==N_Y : # case where the stimulus does not reach limits
    X_0, Y_0 = -.5, 0. # initial position
#    V_X, V_Y = - 2 * X_0 / width , 0. #  target speed
    V_X, V_Y = 1., 0. #  target speed
    loops = .5
else: # centered option, simpler if you think on the torus
    X_0, Y_0 = 0, 0. # initial position
    V_X, V_Y = 1., 0. #  target speed
    loops = 1.
## cloud
sf_0 = 0.15
B_sf = 0.1
B_V = 0.1
B_theta = 10.
## dot
dot_size = .05
## line
radius = .03
if not N_X==N_Y :
    height = .25
else:
    height = .4
falloff = .01
angle = np.pi/4
# pre-processing parameters
white_f_0 = .5
white_alpha = 1.4
white_N = 0.01
#lgn = [1, -.2, -.1, -.05, -.025, -.0125]
#lgn = [1, .5, .25]
#lgn = None
################# particle_grid ##################
if DEBUG: # DEBUG
    N_RF_grid = 5**2
    N_theta_grid = 13
    N_v_grid = 12
else:
    N_RF_grid = 13**2
    N_theta_grid = 17 # for better display, should be a (multiple of 4) + 1
    N_v_grid = 12
v_max_grid = 2.*np.sqrt(V_X**2 + V_Y**2)
log_scale = 1. # base of the logarithmic tiling of particle_grid; linear if equal to one
hex_grid = True
##################  FIGURES   ######################
# parameters for plots
fig_width_pt = 512  # Get this from LaTeX using \showthe\columnwidth / x264 asks for a multiple of 2
dpi = 72.27 # dpi settings to get one point per pixel
inches_per_pt = 1.0/dpi              # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches
fontsize = 10
params = {'backend': 'Agg',
#            'font': "sans-serif",
#           'origin': 'left',
#          'font.family': 'serif',
#          'font.serif': 'Times',
##          'font.sans-serif': 'Times',
#          'text.usetex': True,
#          'mathtext.fontset': 'stix', #http://matplotlib.sourceforge.net/users/mathtext.html
#           'interpolation':'nearest',
          'axes.labelsize': fontsize,
          'font.size': fontsize,
          'legend.fontsize': fontsize,# * 3. /4,
          'xtick.labelsize': fontsize,# * 3. /4,
          'figure.subplot.bottom': 0.12,
          'figure.subplot.left': 0.1,
          'ytick.labelsize': fontsize * 3. /4
        }

figpath = 'results'
matpath = 'data_cache'
for folder in [figpath, matpath]:
    if not(os.path.isdir(folder)):
        try:
            os.mkdir(folder)
        except:
            pass

# ext = '.png' # pdf'
ext = '.png'
vext = '.webm'
vext = '.mp4'
N_show = 2**6 # N_frame # how many particles we show in quiver plots
scale, scale_full = .2 / V_X, 4. / float(N_frame) / V_X # relative scale of vectors resp. in single frames and full quivers
border=0.01 # border between image and figure
line_width = .25# width of the arrows in quiver plots
N_v, N_theta = N_v_grid, N_theta_grid # quantisation for velocity histograms
v_max = v_max_grid # maximal velocity used for readout plots
N_quant_X, N_quant_Y, N_quant_T = 60, None, 100 # quantisation for position histograms
N_quant_X, N_quant_Y, N_quant_T = 60, 1, 100 # quantisation for position histograms
figsize_readout = (4, 4 * np.float(N_Y) / N_X)
#print figsize_readout
#N_first = 1 # if the stimulus is periodic take 0, else take 1
#eps_hist=N_particles*1e-3/N_x/N_y # histogram shows very small values
hue_hist = True # show direction of motion as hue in spatial histogram
hue_time = True # show time hue in particles plot
hue_zoom = False # zooms on the upper right quadrant of motion directions
size_circle =  0.015 # relative size of the circles at the base of the arrows
#figsize_quiver=(fig_width, fig_width * np.float(N_Y) / N_X)
figsize_spat = (fig_width, fig_width*8/13)
figsize_PRF = (fig_width, fig_width*3./4)
figsize_OM = (fig_width, fig_width*3./4)#*(np.sqrt(5)-1.0)/2.)
time_m=0. # smoothing in the OFR model; in units relative to one period
sigma_prior_OM = 0. # std of prior units = speed in the normalized (scaled for a sigma_likelihood of .25 since v_max = 1./(1+(.25/1.)**2) = .95 as in Robinson)
do_errorbars = True
T = 2000. # how long a period lasts (in ms) / it is completely arbitrary
dt= T/float(N_frame) # temporal scale parameter
T_movie = 5. # how long a movie lasts (in s) / it is completely arbitrary
fps = int( N_frame / T_movie) # frame per seconds

# experiment handling parameters
order = 10 # orders of magnitudes for the testing of variables
N_variable = 5
N_step_OM = 4
# statespace parameters
N_blur = 11 # number of value of prediction strength to test
N_noise = 13  # number of value of noise to test
range_blur, range_noise = 1., 1.5 #how far we explore parameters [10^(-range), 10^(range)]

# WARNING : the total number of experiments is N_blur x N_noise x N_trials
N_step = 1 # how much we skip when generating the latex tables for the statespace
X_0_statespace, Y_0_statespace = 0, 0. # initial position
V_X_statespace, V_Y_statespace  = 1., 0. # true speed
N_show_statespace = N_particles / 1 # how many particles we show in quiver plots
N_show_step = 4 # spacing in the LaTeX tables

import pylab
pylab.rcParams.update(params)

# try:
#     # see http://projects.scipy.org/pipermail/scipy-dev/2008-January/008200.html
#     # see https://github.com/rasbt/pyprind
#     # see
#     import pyprind
#     PROGRESS = True
# except:
#     print('pyprind could not be imported')
#     PROGRESS = False
PROGRESS = False

#### particle routines
def condensation(image, N_particles=N_particles, sigma_motion=sigma_motion, K_motion=K_motion,
                 width=width, sigma_noise=sigma_noise, sigma_I=sigma_I, v_prior=v_prior, latency=latency,
                 D_x=D_x, D_V=D_V, resample=resample, bootstrap=bootstrap,
                 v_init=v_init, p_epsilon=p_epsilon,
                 loops=loops, progress=PROGRESS, **kwargs):
    """
    Condensation algorithm.

    :Parameters:

    :Input:
    -------
    a video as a (N_X, N_Y, N_frame) ndarray. corresponds to the whitened luminance profile.

    :Output:
    --------
    `N` particles as a particles.shape array

    :Parameters:
    ------------
    - `N` number of particle batches
    - `D_x`, `D_V` : width of prediction in resp. space and motion domains
    - `resample`
    - `p_epsilon` : the minimal probability in likelihood evaluation
    - [cosmetics] progress (Boolean): whether we show progress

    """
    N_X, N_Y, N_frame = image.shape

    white = whitening(image)
    particles = np.zeros((5, N_particles, np.int(N_frame*loops)))
    if progress :
        pbar = pyprind.ProgBar(int(N_frame*loops), title="Condensation")
    for i_frame in range(0, int(N_frame*loops)):
        # Note: in the CONDENSATION algorithm one may have a different order for the operations (resampling , filtering, prediction). It makes no sense to make the resampling not after the filtering (which sets the weights), so combinations are [filterin, resampling, prediction] or [prediction, filtering, resampling], which are the same cycles of operations. We choose the second one for plotting purposes (showing more arrows with similar weights)
        if i_frame == 0:
            particles_ = prior(N_particles, width=width, v_init=v_init*.5) # HACK
        else:
            ## Going from i_frame-1 to i_frame
            particles_ = particles[:, :, i_frame-1].copy()

        # resampling of particles
        particles_ = resampling(particles_, resample=resample)

        # prediction: P(V_t | I_{0:t-1}) = \int d V_{t-1}  P(V_t | V_{t-1}) P(V_t | I_{t-1})
        # is in the filtering:
        # filtering: P(V_t | I_{0:t}) \propto P(I_t | V_t) P(V_t | I_{0:t-1})
        particles_ = filtering(particles_, white, i_frame, sigma_noise=sigma_noise, sigma_I=sigma_I, sigma_motion=sigma_motion, K_motion=K_motion, p_epsilon=p_epsilon,  width=width, v_prior=v_prior, v_init=v_init)

        particles[:, :, i_frame] = particles_

        if progress : pbar.update()

    particles = push(particles, latency=latency, N_frame=N_frame, D_V=D_V, D_x=D_x, width=width, v_prior=v_prior)

    return particles

def shift_once(particles, N_frame, forwards=True, D_V=D_V, D_x=D_x, width=width, v_prior=v_prior):
    particles_in = particles.copy()
    particles_shifted = np.zeros_like(particles)
    shift = 1
    if not forwards:
        shift = -1
        particles_in[2:4, :, :] *= shift
    for i_frame in range(particles.shape[2]):
        particles_shifted[:, :, i_frame] = prediction(particles_in[:, :, i_frame], N_frame, D_V=D_V, D_x=D_x, width=width, v_prior=v_prior)
    if not forwards:
        particles_shifted[2:4, :, :] *= shift

    particles_shifted = np.roll(particles_shifted, shift, axis=2) # move forwards in time (future) or backwards (past)
    return particles_shifted

def push(particles, latency, N_frame, D_V=D_V, D_x=D_x, width=width, v_prior=v_prior):
    for _ in range(np.abs(latency)):
        particles = shift_once(particles, N_frame, forwards=latency>0, D_V=D_V, D_x=D_x, width=width, v_prior=v_prior)
    return particles

def prediction(particles, N_frame, D_V, D_x, width, v_prior):
    """
    Shifts particles knowing their position and speed.

    A multiplicative noise should favor slow speeds.

    Input
    -----
    N particles : as a particles.shape array
    v_prior : std (in units of velocity) of a prior for slow speeds

    Output
    ------
    N particles as a particles.shape array

    """
    N = particles.shape[1]
    particles_out = particles.copy()
    if particles.ndim is 3:
        for i_frame in range(particles.shape[2]):
            particles_out[:, :, i_frame] = prediction(particles[:, :, i_frame], N_frame, D_V=D_V, D_x=D_x, width=width, v_prior=v_prior)
    else:
        # 1- First, positions are incremented by the velocity (note: converted velocity
        # from relative space to the cortical space metric. Precisely, a velocity
        # of one  V=(1,0) corresponds to one horizontal spatial period in one
        # temporal period, that is, in one frame to a translation of ``width /
        # N_frame`` in cortical space (see RF_grid).
        if not D_V==np.inf:
            particles_out[0:2, :] += particles_out[2:4, :]*width/np.float(N_frame)

        # 2- Then, positions are slightly blurred
        particles_out[0:2, :] += D_x / np.float(N_frame) * np.random.randn(2, N)

        # 3- Finally, the velocity is slightly blurred
        # TODO : implement different D_V for direction and speed
        # todo: implement multiplicative noise with speed = np.sqrt((particles_out[2:4, :]**2).sum(axis=0))
        if not D_V==np.inf:
            if v_prior > 0.:
                particles_out[2:4, :] /= (1 + D_V**2 / v_prior**2)
                particles_out[2:4, :] += (D_V**-2 + v_prior**-2)**-.5 / np.float(N_frame) * np.random.randn(2, N)
            else:
                particles_out[2:4, :] += D_V / np.float(N_frame) * np.random.randn(2, N)
        else:
            particles_out[2:4, :] = prior(N_particles, width=width, v_init=v_init)[2:4, :]

        # make sure to stay on the torus:
        particles_out[0, :] = torus(particles_out[0, :], width)
        particles_out[1, :] = torus(particles_out[1, :], width * np.float(N_Y) / N_X)

    return particles_out

def resampling(particles, resample):
    """
    Resample weighted particles.

    The particles are grouped in N_pop batches to represent multiple velocities.
    The probability is given for all particles but corresponds to the proba-
    bility of one batch.

    Input
    -----
    N particles as a particles.shape array

    Output
    ------
    N particles as a particles.shape array

    Parameters
    ----------
    resample : gives the ratio of particles that get resampled at every frame
                it's 1 in the original CONDENSATION algorithm

    """
    particles_out = particles.copy()
    N = particles.shape[1]  # number of particles

    # TODO : we do not use N_pop anymore, which is some kind of ergodic hypothesis to justify

    # resample a percentage of the particles (the total number keeps to N)
    if not resample==0 and particles_out[4, :].sum()>0:
        # draw resample (in %) random addresses of particles that we will reassign
#         N_resample = np.min((N, np.random.poisson(N*resample))) # using Poisson to be sure to not have rounding errors in case rate < 1
        proba_sum = 0
        while proba_sum==0 : # reassign from the same set
            N_resample = np.random.binomial(N, resample) # using a binomial distribution to be sure to not have rounding errors in case rate < 1
            address_resample = np.random.permutation(np.arange(N))[:N_resample]
            # draw from this subset some addresses uniformly over their pdf using histogram equalization
            proba_resample = particles_out[4, address_resample].copy()
            proba_sum = proba_resample.sum()
        proba_resample /= proba_sum
        address = np.interp(np.linspace(0, 1, N_resample, endpoint=False)+1/2./N_resample,
                            np.concatenate(([0.], np.cumsum(proba_resample))),
                            np.arange(N_resample+1))
        address = [int(k) for k in address]
        # reassign these particles and set their weight to a uniform value
        particles_out[:, address_resample] = particles_out[:, address_resample[address]]
        subset_weight = proba_resample[address].sum() # should be \approx N_resample / N, that is, resample

        particles_out[4, address_resample] = subset_weight / N_resample
        # ultimately, normalize weights
        particles_out[4, :] /= particles_out[4, :].sum()

    return particles_out

def filtering(particles, white, i_frame, sigma_noise=sigma_noise, sigma_I=sigma_I, sigma_motion=sigma_motion, K_motion=K_motion, p_epsilon=p_epsilon,  width=width, v_prior=v_prior, v_init=v_init):
    """
    Filter weighted particles.

    Input
    -----
    N particles as a particles.shape array

    Output
    ------
    N particles as a particles.shape array

    Parameters
    ----------
    sigma_noise, sigma_motion, p_epsilon : see the motion_likelihood function

    """
    N = particles.shape[1]  # number of particles
    particles_at_t = particles.copy()
    particles_at_t_plus_one = prediction(particles_at_t, N_frame, D_V=D_V, D_x=D_x, width=width, v_prior=v_prior)

    likelihood = motion_likelihood(white, particles_at_t_plus_one, i_frame % N_frame, sigma_noise=sigma_noise, sigma_I=sigma_I, sigma_motion=sigma_motion, K_motion=K_motion, p_epsilon=p_epsilon, width=width)
    particles_at_t_plus_one[4, :] = particles_at_t_plus_one[4, :] * likelihood #* prior

    particles_noise = prior(N, width=width, v_init=v_init)
    particles_noise[:2, :] = particles_at_t[:2, :] # HACK ?
    particles_noise[4, :] = particles_at_t[4, :] # HACK ?
    particles_noise = prediction(particles_noise, N_frame, D_V=D_V, D_x=D_x, width=width, v_prior=v_prior)
    likelihood_noise = motion_likelihood(white, particles_noise, i_frame % N_frame, sigma_noise=sigma_noise, sigma_I=sigma_I, sigma_motion=sigma_motion, K_motion=K_motion, p_epsilon=p_epsilon, width=width)
    particles_noise[4, :] = particles_noise[4, :] * likelihood_noise #* prior

    winners_noise = (particles_noise[4, :])  > (particles_at_t_plus_one[4, :])
    particles_at_t_plus_one[:, winners_noise] = particles_noise[:, winners_noise]
    particles_at_t_plus_one[4, :] /= particles_at_t_plus_one[4, :].sum()

    return particles_at_t_plus_one

############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
def show_particles(particles, image=None, N_X=N_X, N_Y=N_Y, N_show=N_show, alpha=1., line_width=line_width,
        normalize=True, color_hue=True, hue_time=hue_time, color=None, fig_width=fig_width, dokey=True, text=None,
        method='mine', axis_label=False, scale=scale, width=width, inset=True, vmin=-1, vmax=1., border=border, fig=None, a=None):
    """
    Shows the quiver plot of a set of particules, optionally associated to an image.

    """
    particles_full = particles.copy()
    # number of particles to show. sorting according to weight
    N = particles.shape[1]
    #print particles.shape
    if N_show < N:
        if particles.ndim == 2: # there is no time
    #        print particles.mean(axis=1)
            particles_show = np.zeros((5, N_show))
            ind = np.argsort(-particles[4, :])
            particles_show = particles[:, ind[:N_show]].copy()
        elif particles.ndim == 3: # there is time
            N_frame = particles.shape[2]
            particles_show = np.zeros((5, N_show, N_frame))

            for i_frame in range(N_frame):
                ind = np.argsort(-particles[4, :, i_frame])
                particles_show[:, :, i_frame] = particles[:, ind[:N_show], i_frame]

        particles = particles_show.copy()
    N = particles.shape[1]


    import matplotlib.cm as cm
    if fig is None: fig = pylab.figure(figsize=(fig_width, fig_width * np.float(N_Y) / N_X))
    # Add an a axes with axes rect [left, bottom, width, height] where all quantities are in fractions of figure width and height.
#    a = fig.add_axes((border, border * np.float(N_Y) / N_X, 1.-2*border, 1.-2*border * np.float(N_Y) / N_X))
    if a is None: a = fig.add_axes((border, border, 1.-2*border, 1.-2*border))#, frameon=False, axisbg='w')
    a.axis(c='b', lw=2)

    ywidth = width * np.float(N_Y) / N_X
    opts = {'vmin':vmin, 'vmax':vmax, 'interpolation':'nearest', 'extent':(-width/2, width/2, -ywidth/2., ywidth/2.), 'aspect':'auto'} # 'origin':'lower',
    if not(image is None):
        if normalize :
            image = .5 + image/np.abs(image).max()/2.
        if image.ndim == 2: opts['cmap'] = pylab.cm.gray
        #  http://matplotlib.sourceforge.net/api/pyplot_api.html?highlight=imshow#matplotlib.pyplot.imshow
        a.imshow(np.swapaxes(image, 0, 1), **opts)
#        a.pcolor(np.linspace(-width/2, width/2, N_X+1), np.linspace(-width/2., width/2., N_Y+1), image, edgecolors='none', linewidths=0, **opts)
    else:
        a.imshow(0.5*np.ones((N_X, N_Y)), **opts)
#        a.pcolor(np.linspace(-width/2, width/2, N_X+1), np.linspace(-width/2., width/2., N_Y+1), 0.5*np.ones((N_X,N_Y)), edgecolors='none', linewidths=0,  **opts)
    a.axis([-width/2, width/2, -ywidth/2., ywidth/2.]) # sets the min and max of the x and y axes, with v = [xmin, xmax, ymin, ymax].

#    print a.axis()


    # computing colors
    if particles.ndim == 2: # there is no time
        Color = np.zeros((N, 4))
        if not(color is None):
            Color[:, 0:3] += np.array(color[0:3], ndmin=2) # R,G,B channels
        else:
            if color_hue:
                for i_N in range(N):
                    sync_angle = -np.pi/6
                    Color[i_N, :] = cm.hsv(np.arctan2(particles[3, i_N]*np.cos(sync_angle)+particles[2, i_N]*np.sin(sync_angle), particles[2, i_N]*np.cos(sync_angle)-particles[3, i_N]*np.sin(sync_angle))/np.pi/2 + .5) # np.arctan2 is in the [np.pi, np.pi] range, cm.hsv takes an argument in [0, 1]
            else:
                Color[:, 0] = 1.
        # weights make a transparency
        Color[:, 3] = particles[4, :].T / particles[4, :].max() * alpha

    elif particles.ndim == 3: # there is time
        N_frame = particles.shape[2]
        if not(color is None):
            print('-/!\ setting a color for a timed particle vector? that''s strange... ', color)
            Color = list()
            for i_N in range(N):
                for i_frame in np.arange(N_frame):
                    Color.append(color[i_frame])
        else:
            Color = list()
            for i_N in range(N):
                for i_frame in range(N_frame):
                    max_weight = particles[4, :, i_frame].max()
                    alpha_ = particles[4, i_N, i_frame] / max_weight *alpha
                    if color_hue:
                        if hue_time: # TODO : there's a better ordering of conditions ...
                            Color.append(cm.hsv(1.*i_frame/N_frame, alpha=alpha_))
                        else:                        # shows orientation as different hues instead of time (confusing)
                            Color.append(cm.hsv(np.arctan2(particles[3, i_N, i_frame]+particles[2, i_N, i_frame],
                                                       particles[2, i_N, i_frame]-particles[3, i_N, i_frame])/np.pi/2 + .5, alpha=alpha_))
                    else:
                        Color.append(cm.hsv(1.*i_frame/N_frame, alpha=alpha_))



    if method == 'mine':
        # using my own quiver with (1) toroidal space (2) segments [X, X+V*scale] (no f**king automatic scaling as in pylab)
        q = my_quiver(a, particles, Color, scale, line_width, base='circle', width=width, ywidth=ywidth)
    else:
        print('not using my method')
        q = a.quiver(particles[0, ...], particles[1, ...],
                 particles[2, ...], particles[3, ...],
                 color=Color, scale=scale, width=0.002*line_width, alpha=alpha)
        if dokey:
            V_max = np.sqrt((particles[2, ...]**2 + particles[3, ...]**2).max())
            V_max = 1# 10**int(log10(max(V_max,.1)))
            cboxh, cboxv = -0.5, -0.89
            boxh, boxv = .2, .05
            a.fill([cboxh - boxh, cboxh + boxh, cboxh + boxh, cboxh - boxh, cboxh - boxh],
                    [cboxv - 2*boxv, cboxv - 2*boxv, cboxv + boxv, cboxv + boxv, cboxv - 2*boxv],
                    edgecolor='k', facecolor='w')
            qk = a.quiverkey(q, cboxh/2 +.5, cboxv/2 +.5, V_max, str(V_max) +' px / frame',
                        alpha=1., color='k', labelpos='S',
                        fontproperties={'weight': 'bold'})

    # TODO : alternatively, adds a text like time step
    # TODO : transform all xlabel(r'\textbf{time (s)}')
    if axis_label : pylab.xlabel(r'X axis'); pylab.ylabel(r'Y axis')
#    else: pylab.axis([-1, 1, -1, 1])
    if inset and  particles.ndim == 2:
#        a_inset = fig.add_axes([.8, .8, .195, .195], axisbg='k', polar=True)
        a_inset = fig.add_axes([0, 0, .3, .3], axisbg='k', polar=True)
        v_hist, v_r_edges, v_theta_edges = vel_readout(particles_full, display=False)
        N_v, N_theta, v_max = v_r_edges.size, v_theta_edges.size, v_r_edges.max()
        [V_r_edges, V_theta_edges] = np.mgrid[v_max/N_v:v_max:1j*N_v,
                (-np.pi+ np.pi/N_theta):(np.pi+ np.pi/N_theta):1j*N_theta]
        a_inset.pcolormesh(V_theta_edges, V_r_edges, v_hist, cmap=pylab.bone(),
                vmin=0., vmax=v_hist.max(), edgecolor='none')
        # draw a cross in the center
        from matplotlib.collections import LineCollection
        # sequence of  r, theta segments (we are in polar coordinates)
        cross = [[(0.,-.2), (.0, .2)],[(np.pi/2, -.2), (np.pi/2, .2)]]
        line_segments = LineCollection(cross, linewidths=.15, colors='w', linestyles='solid')
        a_inset.add_collection(line_segments)
#        cross = [[(0.,-.2), (.0, .2)]]
#        line_segments = LineCollection(cross, linewidths=.05, colors='w', linestyles='solid')
#        a_inset.add_collection(line_segments)
        pylab.setp(a_inset, xticks=[], yticks=[])

    pylab.setp(a, xticks=[], yticks=[])
    pylab.draw()
    return fig, a, q

def my_quiver(a, particles, color, scale, line_width, base, width, ywidth, size_circle = size_circle):
    """

    color contains alpha value
    """
    # HACK for swaping y axis
    X, Y, U, V = particles[0, ...].ravel(), -particles[1, ...].ravel(), particles[2, ...].ravel(), particles[3, ...].ravel()

    if particles.ndim == 3:
        N_frame = particles.shape[2]
    else:
        N_frame = 1

    from matplotlib.collections import LineCollection#, EllipseCollection
    from matplotlib import patches


    def razor(segment):
        # returns True if *all* coordinates of the segment are outside the window
        for value in np.array(segment[:2]).ravel():
            if (abs(value) < width/2.): return False
        for value in np.array(segment[2:]).ravel():
            if (abs(value) < ywidth/2.): return False
        return True

    # draw segments
    seq = list()
    colors = list()
    line_widths = list()

    shifts = [(0., 0.), (width, 0.), (-width, 0.), (0., ywidth), (0., -ywidth), (width, ywidth), (-width, ywidth), (-width, ywidth), (-width, -ywidth)]
#
#    shifts = [(0., 0.), (width, 0.), (-width, 0.), (0., width), (0., -width), (width, width), (-width, width), (-width, width), (-width, -width)]
    for x, y, u, v, color_ in zip(X, Y, U, V, color):
        for x_, y_ in shifts:
            segment = [(x+x_, y+y_), (x+u*scale*width + x_, y+v*scale*width + y_)]
            if not(razor(segment)):
                seq.append(segment)
                colors.append(color_)#[color_[3]*channel for channel in color_])
                line_widths.append(line_width)#*color_[3])

    line_segments = LineCollection(seq, linewidths=line_widths, colors=colors, linestyles='solid')
    a.add_collection(line_segments)

    # draw the arrow bases
    if base == 'angle': # and (particles.ndim < 3):
        centers = list()
        colors = list()
        for x, y, u, v, color_ in zip(X, Y, U, V, color):
            norm = np.sqrt(u**2 + v**2) / line_width * 100.
            centers.append([(x-v/norm, y+u/norm), (x+v/norm, y-u/norm)])
            colors.append(color_)
        arrow_bases = LineCollection(centers, linewidths=line_width, colors=colors, linestyles='solid')
        a.add_collection(arrow_bases)
    elif base == 'circle': # and (particles.ndim < 3):
        for x, y, color_ in zip(X, Y, color):
            # http://matplotlib.sourceforge.net/users/transforms_tutorial.html
            circ = patches.Ellipse((x,y), size_circle / N_frame, size_circle / N_frame, facecolor=color_, edgecolor=color_, alpha=0.5)#
            a.add_patch(circ)

    return a

#### oculo motor read-out
def OM(particles, time_m=time_m, sigma_prior_OM=sigma_prior_OM, N_step=N_step_OM, dt=dt, display=True, figsize=figsize_OM, do_errorbars=do_errorbars):
    """
    Oculo-motor plant.

    Input
    -----
    N particles as a particles.shape array of size (4, N, N_frame)

    Output
    ------
    a tuple of (N_frame)-arrays for the resp. horizontal and vertical traces

    """

#    N = particles.shape[1]
    N_frame = particles.shape[2]
    e_h, e_v = np.zeros(N_frame), np.zeros(N_frame)
    a_h, a_v = np.zeros(N_frame), np.zeros(N_frame)#sigma_a**-2*ones(N_frame), sigma_a**-2*ones(N_frame)
    t_m = time_m * N_frame
    for i_frame in range(N_frame):
        u = particles[2, :, i_frame]
        v = particles[3, :, i_frame]
        w = particles[4, :, i_frame]
        w /= w.sum()

        # TODO: use circular mean and variance for x, y

        # uses the weights to compute the average
        m_h, m_v = (u*w).sum(), (v*w).sum() #u.mean(), v.mean()
        v_h, v_v = ((u - m_h)**2*w).sum(), ((v - m_v)**2*w).sum() # variance along U and V axis

#        if time_v == 0: # Kalman-like adaptation of gaintime_v=3.,
#            a_h[i_frame] = (1-1./time_v)*a_h[i_frame-1] + 1./time_v / v_h
#            a_v[i_frame] = (1-1./time_v)*a_v[i_frame-1] + 1./time_v / v_v
#            time_m_h = time_m * (1 + 1/ (sigma_a**2 * a_h[i_frame-1]) )
#            time_m_v = time_m * (1 + 1/ (sigma_a**2 * a_v[i_frame-1]) )
#        else:
#            time_m_v, time_m_h = time_m, time_m

        if time_m > 0:
            e_h[i_frame] = (1-1./t_m)*e_h[i_frame-1] + 1./t_m*m_h
            e_v[i_frame] = (1-1./t_m)*e_v[i_frame-1] + 1./t_m*m_v
            a_h[i_frame] = (1-1./t_m)*e_h[i_frame-1] + 1./t_m*np.sqrt(v_h)
            a_v[i_frame] = (1-1./t_m)*e_v[i_frame-1] + 1./t_m*np.sqrt(v_v)
        else:
            e_h[i_frame], a_h[i_frame] = m_h, np.sqrt(v_h)
            e_v[i_frame], a_v[i_frame] = m_v, np.sqrt(v_v)

        if sigma_prior_OM > 0:
            e_h[i_frame] = m_h / ( 1 + v_h/sigma_prior_OM**2)
            e_v[i_frame] = m_v / ( 1 + v_v/sigma_prior_OM**2)


    if display:
        time = np.linspace(0., dt*N_frame, N_frame)
        fig = pylab.figure(figsize=figsize)
        a = fig.add_axes((0.15, 0.1, .8, .8))
        if not(do_errorbars):
            a.plot(time, e_h, lw=2, c='r', label='hor')
            a.plot(time, e_v, lw=2, c='b', label='ver')
        else:
            a.errorbar(time[::N_step], e_h[::N_step], yerr=a_h[::N_step], elinewidth=0.5, c='r', label='hor')# fmt='-',
            a.errorbar(time[::N_step], e_v[::N_step], yerr=a_v[::N_step], elinewidth=0.5, c='b', label='ver')
            a.plot(time, e_h, lw=2, c='r', label='hor')
            a.plot(time, e_v, lw=2, c='b', label='ver')
        adjust_spines(a, ['left', 'bottom'])
        pylab.xlabel('time (ms)')
        pylab.ylabel('eye velocity')
        a.legend(loc='right')
        return fig
    else:
        return e_h, e_v, v_h, v_v,a_h,a_v

# def timed_readout(particles, display=True, N_v=12, N_theta=25, v_max=2.):
#     """
#     Reads-out particles into a probability density function.
#
#
#     Input
#     -----
#     N particles as a particles.shape array
#
#     Output
#     ------
#     pdf
#
#     """
#
# #    N = particles.shape[1]
#     N_frame = particles.shape[2]
#
#     v_hist, v_r_edges, v_theta_edges = vel_readout(particles[:, :, 0], display=False, N_v=N_v, N_theta=N_theta, v_max=v_max)
#     timed_hist = np.zeros(v_r_edges.shape, v_theta_edges.shape, N_frame)
#     timed_hist[:, :, 0] = v_hist
#
#     for i_frame in range(1, N_frame):
#         v_hist, v_r_edges, v_theta_edges = readout(particles[:, :, i_frame], display=False, N_v=N_v, N_theta=N_theta, v_max=v_max)
#         timed_hist[:, :, i_frame] = v_hist
#
#     return timed_hist, v_r_edges, v_theta_edges
#
def vel_readout(particles, N_v=N_v, N_theta=N_theta, v_max=v_max, display=True, direction=False, figsize=figsize_readout):
    """
    Reads-out particles into a probability density function in velocity space.

    Marginalization over all positions.

    Input
    -----
    N particles as a particles.shape array

    Output
    ------
    a velocity PDF

    """

    u = particles[3, ...].ravel()
    v = particles[2, ...].ravel()
    # we weight the readout by the weight of the particles
    weights = particles[4, ...].ravel()

    v_r = np.sqrt(u**2 + v**2)
    v_theta = np.arctan2(u*np.cos(np.pi/N_theta)-v*np.sin(np.pi/N_theta),
                         v*np.cos(np.pi/N_theta)+u*np.sin(np.pi/N_theta))

    v_r_edges = np.linspace(0, v_max, N_v)
    v_theta_edges = np.linspace(-np.pi, np.pi, N_theta)
    v_hist, v_r_edges_, v_theta_edges_ = np.histogram2d(v_r, v_theta, (v_r_edges, v_theta_edges), normed=True, weights=weights)
    # still, it is normalized in polar coordinates: we should divide the density by the area of the wedge:
    v_r_edges_middle = .5*(v_r_edges_[:-1] + v_r_edges_[1:])
    v_hist /= v_r_edges_middle[:, np.newaxis]
    v_hist /= v_hist.sum()

    if display:
        fig = pylab.figure(figsize=figsize)
        a = pylab.axes([0.1, 0.1, 0.8, 0.8], polar=True, axisbg='w')
        if direction:
            theta = .5*(v_theta_edges[:-1]+v_theta_edges[1:])
            prob = v_hist.sum(axis=0)
            a.plot(np.concatenate((theta, theta[0:1] + 2*np.pi)), np.concatenate((prob, prob[0:1])))
            pylab.setp(a, yticks=[])

        else:
            [V_r_edges, V_theta_edges] = np.mgrid[v_max/N_v:v_max:1j*N_v,
                        (-np.pi+ np.pi/N_theta):(np.pi+ np.pi/N_theta):1j*N_theta]
#             [V_r_edges, V_theta_edges] = mgrid[v_max/N_v:v_max:1j*N_v, -pi:pi:1j*N_theta]
            a.pcolormesh(V_theta_edges, V_r_edges, v_hist, cmap=pylab.bone(), vmin=0., vmax=v_hist.max())#, edgecolor='k')
        return fig
    else:
        return v_hist, v_r_edges, v_theta_edges

def spatial_readout(particles, N_X=N_X, N_Y=N_Y, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
        display=True, fig=None, a=None, ruler=True,
        hue=hue_hist, hue_zoom=hue_zoom, fig_width=fig_width, width=width):

    """
    Reads-out particles into a probability density function in spatial space.

    Instead of a quiver plot, it makes an histogram of the density of particles
    by: 1) transforming a particle set in a 3 dimensional (x,y, \theta) density
    (let's forget about speed norm), (2) showing direction spectrum as hue and
    spatial density as transparency

    Marginalization over all speeds.

    Input
    -----
    N particles as a particles.shape array

    Output
    ------
    a position PDF

    """

    if N_quant_Y is None:
        N_quant_Y = int(N_quant_X * np.float(N_Y) / N_X) # quantisation for position histograms
    N_frame = None
    if particles.ndim == 3: N_frame = particles.shape[2]
    if (N_quant_Y is 1) and not (N_frame is None) :
        v_hist = np.zeros((N_quant_X, N_quant_T))
        v_hist_u = np.zeros((N_quant_X, N_quant_T))
        x_edges = np.linspace(-width/2, width/2, N_quant_X+1)
        u_edges = np.linspace(-1.5, 1.5, N_quant_X+1)
        for t in range(N_quant_T) :
            t_range_b = int(t*N_frame/N_quant_T)
            t_range_e = int((t*N_frame+N_frame)/N_quant_T)
            x = particles[0, :, t_range_b:t_range_e].ravel()
            u = particles[2, :, t_range_b:t_range_e].ravel()
            # we weight the readout by the weight of the particles
            weights = particles[4,  :, t_range_b:t_range_e].ravel()
            v_hist[:, t], x_edges_ = np.histogram(x, x_edges, normed=False, weights=weights)
            v_hist_u[:, t], u_edges_ = np.histogram(u, u_edges, normed=False, weights=weights)
        y_edges = None
    else:
        x = particles[0, ...].ravel()
        y = particles[1, ...].ravel()
        # we weight the readout by the weight of the particles
        weights = particles[4, ...].ravel()
        # ywidth = width * np.float(N_Y) / N_X
        x_edges = np.linspace(-width/2, width/2, N_quant_X+1)
        y_edges = np.linspace(-width/2 * np.float(N_Y)/N_X, width/2 * np.float(N_Y)/N_X, N_quant_Y+1)
        if hue and display and not (N_quant_Y is 1):
            N_theta_=3 # the 3 RGB channels
            u = particles[3, ...].ravel()
            v = particles[2, ...].ravel()
            v_theta = np.arctan2(u+v, v-u)
            if hue_zoom:
                v_theta_edges = np.linspace(-np.pi/4-np.pi/8, -np.pi/4+np.pi/8, N_theta_ + 1 )
            else:
                v_theta_edges = np.linspace(-np.pi, np.pi, N_theta_ + 1 )# + pi/N_theta

            sample = np.hstack((x[:,np.newaxis], y[:,np.newaxis], v_theta[:,np.newaxis]))
            bin_edges = (x_edges, y_edges, v_theta_edges)
            v_hist, edges_ = np.histogramdd(sample, bins=bin_edges, normed=True, weights=weights)
            v_hist /= v_hist.sum()
        else:
            v_hist, x_edges_, y_edges_ = np.histogram2d(x, y, (x_edges, y_edges), normed=True, weights=weights)
            v_hist /= v_hist.sum()

    if display:
        if N_quant_Y is 1 and (N_frame is None)  : # PDF of x at some time
            if fig is None: fig = pylab.figure(figsize=(fig_width, fig_width * np.float(N_Y) / N_X))
            if a is None:
                a = fig.add_subplot(1, 1, 1)
                a.axis(c='b', lw=2)
            a.plot(.5*x_edges[:-1] + .5*x_edges[1:], v_hist.ravel())
            a.set_ylim([0., v_hist.max()])
#             a.axis([-width/2, width/2, -width/2 * np.float(N_Y)/N_X, width/2 * np.float(N_Y)/N_X])
        elif N_quant_Y is 1 and not (N_frame is None)  : # X-T plot
            if fig is None: fig = pylab.figure(figsize=(fig_width * np.float(N_quant_T) / N_X, fig_width))
#             if a is None: a = fig.add_subplot(1, 1, 1)
            if a is None:
                a = []
                a.append(fig.add_axes([0.1, 0.07, .86, .41]))
                a.append(fig.add_axes([0.1, 0.57, .86, .41]))
                for ax in a: ax.axis(c='b', lw=2)
#             time, (.5*x_edges[:-1] + .5*x_edges[1:]), print (time[np.newaxis, :].shape)
            x_middle = .5*(x_edges[1:] + x_edges[:-1])
            v_hist /= v_hist.max(axis=0)[np.newaxis, :]
            Time, X = np.meshgrid(np.linspace(0, 1, N_quant_T), x_edges_)# .5*(x_edges_[:-1]+x_edges_[1:]))
            a[0].pcolormesh(Time, X, v_hist, cmap=pylab.cm.Blues, vmin=0., vmax=v_hist.max())#, edgecolor='k')#(.1, .1, .1, .6))
            Time, U = np.meshgrid(np.linspace(0, 1, N_quant_T), u_edges_)# .5*(u_edges_[:-1]+u_edges_[1:]))
            v_hist_u /= v_hist_u.max(axis=0)[np.newaxis, :]
            a[1].pcolormesh(Time, U, v_hist_u, cmap=pylab.cm.Reds, vmin=0., vmax=v_hist_u.max())#, edgecolor='k')#(.1, .1, .1, .6))
            if ruler:
                a[0].plot([0, 1], [-1, 1], 'b--', lw=2)
                for i, min_y, max_y in zip(range(2), [-1, -1.4], [1, 1.4]):
                    for x, c in zip([.2, .3, .5, .6, .8, .9], ['k', 'g', 'k', 'g', 'k', 'g']):
                        a[i].plot([x, x], [min_y, max_y], ls='--', lw=2, c=c)
                a[1].plot([0, 1], [0, 0], 'r--', lw=2)
                a[1].plot([0, 1], [1, 1], 'r--', lw=2)
#             a[0].axis([0, N_frame, 0, N_quant_X])
#             a[1].axis([0, N_frame, 0, N_quant_X])
            a[0].set_xlabel('Time (s)')
            a[0].set_ylabel('Space')
            a[1].set_ylabel('Velocity')
        else:  # PDF of (x, y) at some time
#             print('2D pdf', N_X, N_Y, N_quant_X, N_quant_Y,         display, fig, a,         hue, hue_zoom, fig_width, width)
            if fig is None: fig = pylab.figure(figsize=(fig_width, fig_width * np.float(N_quant_Y) / N_quant_X))
            if a is None:
                a = fig.add_axes([0., 0., 1., 1.])
                a.axis(c='b', lw=2)

            if hue :
                # TODO : overlay image and use RGB(A) information
                a.imshow(np.fliplr(np.rot90(v_hist/v_hist.max(),3)), interpolation='nearest', origin='lower',
                        extent=(-width/2, width/2, -width/2 * np.float(N_quant_Y)/N_quant_X, width/2 * np.float(N_quant_Y)/N_quant_X))#
            else:
                a.pcolormesh(x_edges, y_edges, v_hist.T, cmap=pylab.bone(), vmin=0., vmax=v_hist.max())#, edgecolor='k')
            a.axis([-width/2, width/2, -width/2 * np.float(N_quant_Y)/N_quant_X, width/2 * np.float(N_quant_Y)/N_quant_X])

        return fig, a
    else:
        return v_hist, x_edges, y_edges

def PRF(parameter, gain, text='contrast', figsize=figsize_PRF, order=order):
    """
    Shows the parameter response function (by default the contrast respone
    function or CRF).

    """
    # TODO make a ---O---O--- plot
    fig = pylab.figure(figsize=figsize)
    a = fig.add_axes((.1, .1, .88, .88))
    a.errorbar(parameter, gain[0, :].mean(axis=1), yerr=gain[0, :].std(axis=1),
               lw=2, c='r', label='hor')
    a.errorbar(parameter, gain[1, :].mean(axis=1), yerr=gain[1, :].std(axis=1),
               lw=2, c='b', label='ver')

    adjust_spines(a, ['left', 'bottom'])
    pylab.xlabel(text)
    pylab.ylabel('gain')
    a.legend()#loc = 'left')
    if not order is None:
        if order > 1:
            a.set_xscale('log')
    pylab.draw()
    return fig, a


#### low-level velocity
def particle_grid(N_v=N_v_grid, N_theta=N_theta_grid, v_max=v_max_grid,
                  N_RF=N_RF_grid, width=width, log_scale=log_scale,
                  hex_grid=hex_grid, ratio=1. * np.float(N_Y) / N_X):
    """
    Creates a set of velocity samples on a radial grid x position samples on a
    rectangular (hexagonal) grid. This is useful to show the likelihood of the
    optical flow.

    By convention, velocity on the torus is such that V=(1,0) corresponds to
    one horizontal spatial period in one temporal period.
    Knowing the relevent parameters, this implies that in one frame, a
    translation is of :
    - ``width / N_frame`` in cortical space.
    -  ``N_X / width / N_frame`` pixels in image space (see translate and
    MotionEnergy).

    Parameters for the log_scale option: see Stocker

    """
    if log_scale > 1. :
        v_rho = np.logspace(np.log(v_max/N_v)/np.log(log_scale),
                            np.log(v_max)/np.log(log_scale), num=N_v,
                            endpoint=True, base=log_scale)
    else:
        v_rho = v_max - np.linspace(0, v_max, N_v, endpoint=False)
    if hex_grid:
        parity = np.arange(N_v) % 2

    v_theta = np.linspace(0, 2*np.pi, N_theta, endpoint=False)

    # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of N_RF**2 dots?"
    N_RF_X = np.int(np.sqrt(N_RF*np.sqrt(3)))
    N_RF_Y = np.int(np.sqrt(N_RF/np.sqrt(3)))
    RF = np.zeros((2, N_RF_X*N_RF_Y))
    X, Y = np.mgrid[-1:1:1j*(N_RF_X+1), -1:1:1j*(N_RF_Y+1)]

    # It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
    X, Y = X[1:, 1:], Y[1:, 1:]
    # Add to every even Y a half RF width to generate hex grid
    Y[::2, :] += (Y[0, 0] - Y[0, 1])/2 # 1./N_RF
    RF[0, :] = X.ravel() * width/2
    # TODO do a tiling in the rectangular case too?
    RF[1, :] = Y.ravel() * width/2 * ratio


    particles = np.ones((5, N_v * N_theta * N_RF))
    index = 0
    for i_v_rho, rho in enumerate(v_rho):
        for i_theta, theta in enumerate(v_theta):
            for i_RF in range(N_RF_X*N_RF_Y):

                particles[0, index] = RF[0, i_RF]
                particles[1, index] = RF[1, i_RF]
                particles[2, index] = np.cos(theta + parity[i_v_rho] * np.pi / N_theta) * rho
                particles[3, index] = np.sin(theta + parity[i_v_rho] * np.pi / N_theta) * rho
                particles[4, index] = 1. / (N_v * N_theta * N_RF) # weights are equal
                index += 1

    return particles

def proba_signal(I, sigma_noise, sigma_I, p_s):
    p = np.exp( I**2 / 2 / sigma_noise**2 / (sigma_noise**2 /sigma_I**2 + 1) )
    p /= np.sqrt(sigma_I**2 /sigma_noise**2 + 1)
    if p_s<1:
        p /= 1/p_s - 1
        return 1 / (1 + 1/p)
    else:
        return 1

def motion_likelihood(image, particles, frame, sigma_motion, K_motion, width, sigma_noise, sigma_I, p_epsilon, corr=True):
    N_X, N_Y, N_frame = image.shape
    ywidth = width * np.float(N_Y) / N_X
    x, y, u, v, w = particles#[0:4, :]
    N = particles.shape[1]
    x_px = (x + width/2) / width * np.float(N_X)
    y_px = (y + ywidth/2) / ywidth * np.float(N_Y)
    I_pre =  image[x_px.astype(np.int), y_px.astype(np.int), frame]
    x_px_pred = np.mod(x_px + u*np.float(N_X)/N_frame, N_X)
    y_px_pred = np.mod(y_px + v*np.float(N_Y)/N_frame, N_Y)
    sigma_motion_ = sigma_motion * np.sqrt(1 + K_motion**2 * (u**2 + v**2))
    I_post = image[x_px_pred.astype(np.int), y_px_pred.astype(np.int), (frame+1) % N_frame]
    I_err2 = (I_post - I_pre)**2
    likelihood = np.exp(-.5*I_err2/sigma_motion_**2)
    likelihood /= np.sqrt(2*np.pi) * sigma_motion_
    likelihood *= proba_signal(I_post, sigma_noise, sigma_I, p_epsilon)

    if likelihood.sum() == 0:
        print('/!\ computed a null probability vector ')
        return np.ones((N))*1./N
    else:
        return likelihood

def prior(N, width, v_init, polar_init=polar_init):
    """"
    Returns N samples drawn from a standard prior

    - uniform in space, normal in speed
    - TODO http://en.wikipedia.org/wiki/V._A._Epanechnikov

    """
    if polar_init :
        v_theta = np.random.rand(N)*2*np.pi
        v_rho = np.random.randn(N)*v_init
        u = np.cos(v_theta) * v_rho
        v = np.sin(v_theta) * v_rho
        return np.vstack((np.random.rand(N)*width-width/2.,
                  np.random.rand(N)*width-width/2.,
                  u, v,
                  1.*np.ones((N))/float(N)))
    else:
        return np.vstack((np.random.rand(N)*width-width/2.,
                      np.random.rand(N)*width-width/2.,
                      np.random.randn(N)*v_init,
                      np.random.randn(N)*v_init,
                      1.*np.ones((N))/float(N)))

#### image processing routines
def whitening_filt(size, temporal=False, f_0=white_f_0, alpha=white_alpha, N=white_N):
    """
    Returns the whitening filter.

    Uses the low_pass filter used by (Olshausen, 98) where
    f_0 = 200 / 512

    parameters from Atick (p.240)
    f_0 = 22 c/deg in primates: the full image is approx 45 deg
    alpha makes the aspect change (1=diamond on the vert and hor, 2 = anisotropic)

    """
    # TODO: make range from -.5 to .5
    fx, fy, ft = np.mgrid[-1:1:1j*size[0], -1:1:1j*size[1], -1:1:1j*size[2]]
    if temporal:
        rho = np.sqrt(fx**2+ fy**2 + ft**2) #TODO : test the effect of whitening parameters? seems to leave a trail... + acausal
    else:
        rho = np.sqrt(fx**2+ fy**2)
    low_pass = np.exp(-(rho/f_0)**alpha)
    K = (N**2 + rho**2)**.5 * low_pass
    return  K

def FTfilter(image, FTfilter):
    from scipy.fftpack import fftn, fftshift, ifftn, ifftshift
    from scipy import real
    FTimage = fftshift(fftn(image)) * FTfilter
    return real(ifftn(ifftshift(FTimage)))

def whitening(image):
    """
    Returns the whitened sequence
    """
    K = whitening_filt(size=image.shape)
    white = FTfilter(image, K)
#     if not lgn is None :
#         white_ = white.copy()
#         white *= 0.
#         for i, a in enumerate(lgn):
#             white += a*np.roll(white_, i, axis=-1)
    return white

def torus(x, w):
    """
    center x in the range [-w/2., w/2.]

    To see what this does, try out:
    >> x = np.linspace(-4,4,100)
    >> pylab.plot(x, torus(x, 2.))

    """
    return np.mod(x + w/2., w) - w/2.

def translate(image, vec):
    """
    Translate image by vec (in pixels)

    """
    u, v = vec

    if image.ndim == 2:
        # first translate by the integer value
        image = np.roll(np.roll(image, np.int(u), axis=0), np.int(v), axis=1)
        u -= np.int(u)
        v -= np.int(v)

        # sub-pixel translation
        from scipy import mgrid
        f_x, f_y = mgrid[-1:1:1j*image.shape[0], -1:1:1j*image.shape[1]]
        trans = np.exp(-1j*np.pi*(u*f_x + v*f_y))
        return FTfilter(image, trans)

    else:
        # TODO : one shot 3D FFT?
        image_ = np.zeros(image.shape)
        for i_frame in range(image.shape[2]):
            image_[:, :, i_frame] = translate(image[:, :, i_frame], vec)
        return image_


#### plotting routines
def anim_save(z, filename, N_X=N_X, N_Y=N_Y, N_quant_X=None, N_quant_Y=None, particles=None, N_show=N_show, display=False, normalize=True, hue=True, vext=vext,
        first_frame=False, last_frame=False, dpi=dpi, fps=fps, line_width=line_width, progress=PROGRESS, scale=scale, loops=loops, verbose=False):
    """
    see http://web.media.mit.edu/~lifton/snippets/graph_movie/GraphMovieDemo.py

    """
    import tempfile
    import imageio
    def make_frames(z, particles, N_X=N_X, N_Y=N_Y, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y):
        if not(z is None): #
            N_X_, N_Y_, N_frame = z.shape
            if not(particles is None):
                N_frame_particles = particles.shape[2]
                loops_ = int(N_frame_particles/N_frame)
            else:
                loops_ = 1
            if normalize:
                z = (.5* z/np.abs(z).max() + .5)
            else:
                z = (.5* z + .5)
        elif not(particles is None):
            # there is no image, so we take as many steps as there are in the particle vector
            N_frame = particles.shape[2]
            loops_ = 1
        else:
            print('Nothing to show!')
        files = []
        tmpdir = tempfile.mkdtemp()

        print('Saving sequence ' + filename + vext)
        if progress: pbar = pyprind.ProgBar(int(N_frame*loops), title="Saving movie")
        for i_frame in range(int(N_frame*loops)):
            if PROGRESS: pbar.update()#i_frame)
            fname = os.path.join(tmpdir, 'frame%03d.png' % i_frame)
            # print 'Saving frame', fname
            if (particles is None): # no particle vector
                im_ = z[:, :, i_frame % N_frame]
                im_ = np.fliplr(im_).T
#                 toimage(im_, cmin=0., cmax=1.).save(fname)
                imageio.imwrite(fname, im_)

            else: # TODO: use a switch to choose the type of visualization?
                fig_width = 10
                fig = pylab.figure(figsize=(fig_width, fig_width * np.float(N_Y) / N_X))
                if (z is None): # TODO : this assumes spatial_readout cannot do a imshow
                    fig, a = spatial_readout(particles[:, :, i_frame], N_X=N_X, N_Y=N_Y, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y, hue=hue, fig=fig)
                else:
#                    print z.min(), z.max()
                    fig, a, q = show_particles(particles[:, :, i_frame], image=z[:, :, i_frame % N_frame], N_X=N_X_, N_Y=N_Y_, N_show=N_show, line_width=line_width, scale=scale, normalize=False, fig=fig)
                pylab.draw()
                fig.savefig(fname, dpi=256./fig_width) #
                pylab.close(fig)
            files.append(fname)
        return tmpdir, files

    def remove_frames(tmpdir, files):
        """
        Remove frames from the temp folder

        """
        for fname in files: os.remove(fname)
        if not(tmpdir is None): os.rmdir(tmpdir)
    if verbose:
        verb_ = ''
    else:
        verb_ = ' 2>/dev/null'
    if not(os.path.isfile(filename + vext)):
        # 1) create temporary frames
        tmpdir, files = make_frames(z, particles=particles, N_X=N_X, N_Y=N_Y, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y)

        # 2) convert frames to movie
        if vext == '.mpg':
            os.system('ffmpeg -v 0 -y  -f image2 -r ' + str(fps) + ' -sameq -i ' + tmpdir + '/frame%03d.png  ' + filename + vext + verb_)

        if vext == '.mp4': # specially tuned for iPhone/iPod http://www.dudek.org/blog/82
    #        options = '-v 0 -y -f mp4 -sameq -vcodec libx264 -maxrate 1000 -b 700 -bufsize 4096 -g 300 -acodec aac -ab 192i -loop_output 0 -s 320x320 -v 0 -y '
            #         options = '-f mp4 -sameq -vcodec libx264 -maxrate 1000k -b 700k -bufsize 4096k -g 300 -v 0 -y '
    #        options = ' -pix_fmt yuv420p -vcodec libx264 -y '
#             options = ' -f mp4 -pix_fmt yuv420p -c:v libx264  -g ' + str(fps) + '  -r ' + str(fps) + ' -y '
            options = ' -f mp4 -pix_fmt yuv420p -vcodec libx264  -g ' + str(fps) + '  -r ' + str(fps) + ' -y '

            os.system('ffmpeg -i '  + tmpdir + '/frame%03d.png  ' + options + filename + vext + verb_)

        # # use https://pypi.python.org/pypi/numpngw ?
        # write_apng("example5.png", seq, delay=50, use_palette=True)

        if vext == '.gif': # http://www.uoregon.edu/~noeckel/MakeMovie.html
            # 2) convert frames to movie
            #for convert, options = ' -set delay 8 -colorspace GRAY -colors 256 -dispose 1 -loop 0 ' #
            options = ' -pix_fmt rgb24 -r ' + str(fps) + ' -loop_output 0 '
            os.system('ffmpeg -i '  + tmpdir + '/frame%03d.png  ' + options + filename + vext + verb_)

        if vext == '.webm':
            # 2) convert frames to movie
            options = ' -f webm  -pix_fmt yuv420p -vcodec libvpx -qmax 12 -g ' + str(fps) + '  -r ' + str(fps) + ' -y '
            cmd = 'ffmpeg -i '  + tmpdir + '/frame%03d.png ' + options + filename + vext + verb_
            os.system(cmd)

        if first_frame:# eventually saves first frame
            os.system('cp ' + os.path.join(tmpdir, files[0]) + ' ' + filename + vext.replace('.', '_') + '_first.png')
        if last_frame:# eventually saves first frame
            os.system('cp ' + os.path.join(tmpdir, files[-1]) + ' ' + filename + vext.replace('.', '_') + '_last.png')

        # 3) clean up
        remove_frames(tmpdir, files)
    else:
        print('Sequence ' + filename + vext + ' already exists (remove to recompute)')


    if display:
        if not(vext == '.mp4'): raise('can only display a mpg file')
#        else: os.system('mplayer -fs ' + filename + vext + '  -loop 0')
        else: os.system('open ' + filename + vext )

def adjust_spines(ax, spines):
    """ a small function to draw plot axis properly, that is not (joined and forming a box)
    """
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10)) # outward by 10 points
        else:
            spine.set_color('none') # don't draw spine


    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

import os
def touch(fname):
    open(fname, 'w').close()

def mat_condensation(matname, image=None, loops=loops, recompute=recompute, **kwargs):
    """
    Returns the state of an experiment - computes CONDENSATION if needed.

    """
    if not(os.path.isfile(matname)) or recompute:
        if  image is None: # just checkin', no computing please
            if not(os.path.isfile(matname + '_lock')):
                return 'todo' #not done, not locked... trying to do with None image?'
            else:
                return 'lock'
        else:
            time.sleep(np.random.rand()*sleeping_time)
            if not(os.path.isfile(matname + '_lock')):
                touch(matname + '_lock')
                touch(matname + LOCK)
                particles = condensation(image, loops=loops, **kwargs)
                np.save(matname, particles)
                try:
                    os.remove(matname + LOCK)
                    os.remove(matname + '_lock')
                except:
                    print('Coud not remove ', matname + LOCK)
                return 'done' # just did it
            else:
                return 'lock' # not finished in another process
    else:
        return 'done' # already done

#### experiment handling
def figure_condensation(figname, image, loops=loops, particles=None, ext=ext, N_show=N_show, figures=True,
                        N_quant_X=N_quant_X, N_quant_Y=N_quant_Y, scale=scale, scale_full=scale_full, video=True, dpi=dpi,
                        vext=vext, direction=False, line_width=line_width, **kwargs):

    switch_locked = False
    N_X, N_Y, N_frame = image.shape
    if (particles is None): # in some cases, particles are already computed and directly passed to the function, in general, we do the computation (or retrieve it from cached files in the matpath folder)
        matname = figname.replace(figpath, matpath) + '.npy'
        switch_locked = mat_condensation(matname, image, loops=loops, **kwargs)

    if switch_locked == 'lock':
        print(matname, ' is locked !')
    else:
        try: # tidy up
            os.remove(matname + LOCK)
            os.remove(matname + '_lock')
        except:
            pass
        try:#if True:#
            if (particles is None): particles = np.load(matname)
#             figname_ = figname + '_first' + ext
#             if not(os.path.isfile(figname_)) and not(os.path.isfile(figname_ + '_lock')) and figures:
#                 touch(figname_ + '_lock')
#                 touch(figname_ + LOCK)
#                 fig, a, q = show_particles(particles[:, :, 0], image=image[:, :, 0], N_X=N_X, N_Y=N_Y, N_show=N_show, line_width=line_width, scale=scale)
#                 fig.savefig(figname_)
#                 pylab.close(fig)
#                 os.remove(figname_ + LOCK)
#                 os.remove(figname_ + '_lock')

            figname_ = figname + '_particles' + ext
            if not(os.path.isfile(figname_)) and not(os.path.isfile(figname_ + '_lock')) and figures:
                touch(figname_ + '_lock')
                touch(figname_ + LOCK)
                fig, a, q = show_particles(particles[:, :, :], N_X=N_X, N_Y=N_Y, N_show=N_show, line_width=line_width, scale=scale_full)
                fig.savefig(figname_)
                pylab.close(fig)
                os.remove(figname_ + LOCK)
                os.remove(figname_ + '_lock')

#             figname_ = figname + '_last' + ext
#             if not(os.path.isfile(figname_)) and not(os.path.isfile(figname_ + '_lock')) and figures:
#                 touch(figname_ + '_lock')
#                 touch(figname_ + LOCK)
#                 fig, a, q = show_particles(particles[:, :, int(N_frame*loops)-1], image=image[:, :, int(N_frame*loops)-1], N_X=N_X, N_Y=N_Y, N_show=N_show, line_width=line_width, scale=scale)
#                 fig.savefig(figname_)
#                 pylab.close(fig)
#                 os.remove(figname_ + LOCK)
#                 os.remove(figname_ + '_lock')
#     #
    #        figname_ = figname + '_vel_readout_init' + ext
    #        if not(os.path.isfile(figname_)) and not(os.path.isfile(figname_ + '_lock')) and figures:
    #            touch(figname_ + '_lock')
    #            touch(figname_ + LOCK)
    #            fig = vel_readout(particles[:, :, 0], direction=direction)
    #            fig.savefig(figname_)
    #            pylab.close(fig)
    #            os.remove(figname_ + LOCK)
    #            os.remove(figname_ + '_lock')
    #        figname_ = figname + '_vel_readout' + ext
    #        if not(os.path.isfile(figname_)) and not(os.path.isfile(figname_ + '_lock')):
    #            touch(figname_ + '_lock')
    #            touch(figname_ + LOCK)
    #            fig = vel_readout(particles[:, :, -1], direction=direction)
    #            fig.savefig(figname_)
    #            pylab.close(fig)
    #            os.remove(figname_ + LOCK)
    #            os.remove(figname_ + '_lock')

            figname_ = figname + '_spatial_readout' + ext
            if not(os.path.isfile(figname_)) and not(os.path.isfile(figname_ + '_lock')) and figures:
                touch(figname_ + '_lock')
                touch(figname_ + LOCK)
                #fig,a = spatial_readout(particles[:, :, -1], N_X=N_X, N_Y=N_Y, N_quant_X=N_quant_X, N_quant_Y=1, width = width)
                fig,a = spatial_readout(particles, N_X=N_X, N_Y=N_Y, N_quant_X=N_quant_X, N_quant_Y=1, width = width)
                fig.savefig(figname_)
                pylab.close(fig)
                os.remove(figname_ + LOCK)
                os.remove(figname_ + '_lock')

            figname_ = figname + '_spatial_readout' + vext
            if not(os.path.isfile(figname_)) and not(os.path.isfile(figname_ + '_lock')) and video:
                touch(figname_ + '_lock')
                touch(figname_ + LOCK)
                anim_save(z=None, filename=figname + '_spatial_readout', N_X=N_X, N_Y=N_Y, particles=particles, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y, loops=1, dpi=dpi, vext=vext)
                pylab.close('all')
                os.remove(figname_ + LOCK)
                os.remove(figname_ + '_lock')

            figname_ = figname + '_particles' + vext
            if not(os.path.isfile(figname_)) and not(os.path.isfile(figname_ + '_lock')) and video:
                touch(figname_ + '_lock')
                touch(figname_ + LOCK)
                anim_save(image, filename=figname + '_particles', N_X=N_X, N_Y=N_Y, particles=particles, N_show=N_show, loops=loops, dpi=dpi, vext=vext, line_width=line_width, scale=scale)
                pylab.close('all')
                os.remove(figname_ + LOCK)
                os.remove(figname_ + '_lock')
        except Exception as e:
            print('figures could not be plotted', e)
    return switch_locked

def make_figname(figname, new_kwargs, i_trial=None):
    figname_ = figname
    for key in np.sort(list(new_kwargs.keys())):
        # print(key, new_kwargs[key])
        figname_ += '-' + str(key) + '_' + ( '%.4f' % new_kwargs[key]).replace('.', '_')
    if not i_trial is None: figname_ += '-trial_' + str(i_trial)
    return figname_

def figure_image_variable(figname, N_X, N_Y, N_frame, generate, ext=ext, N_variable=N_variable, order=order,
                          N_trials=N_trials, loops=loops, make_legend=True, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                          do_figures=False, do_videos=False, do_figure=True, do_video=True, figures_OFR=True, T=T,
                          fixed_args={}, **kwargs):
    """
    TODO: documentation

    """

    print("Studying CONDENSATION by changing image along variable(s) :", kwargs)
    if (order is None): # means we specified the variables directly in a vector
        N_variable_check = None
        for key in kwargs:
            #print(key)
            variable = kwargs[key]
            if N_variable_check is None: N_variable_check = len(variable)
            N_variable = len(variable)
            # check length of different N_variable
            if not(N_variable_check==N_variable): print("there are multiple variables changing and all vectors must be of the same length, here comes trouble! ")
    elif (order==1):
        print(' order is one, you may have problems scaling your variable...!')

    # ordering everything in a list of dictionaries
    kwargs_variable = [] # holds a list of the variables we test, each element being a dictionary having as keys the given variable and as a value the value of the variable that we test
    #shuffling = np.random.permutation(np.arange(N_variable)) # shuffle the order in which the experiments are being done (but not the results!)
    for i_variable in range(N_variable): #shuffling:
        kwarg_tmp = {}
        for key in kwargs:
            if (order is None):
                kwarg_tmp[key] = kwargs[key][i_variable]
                if type(kwarg_tmp[key]) is np.ndarray: kwarg_tmp[key] = kwarg_tmp[key].tolist()
            elif order > 0: # order gives the exponent by which the variable is scaled
                kwarg_tmp[key] = order**( (N_variable-1-2*i_variable) / (N_variable-1)) * kwargs[key] #
            else: # case when order = 0 and the range is linear
                kwarg_tmp[key] = (i_variable+1.)/ N_variable * kwargs[key] #
        kwargs_variable.append(kwarg_tmp)

    # doing the experiments
    for stage in ['running all scripts', 'walking again to see if a parallel run did not finish the job']:
        switch_finished = True
        for new_kwargs in np.random.permutation(kwargs_variable):
            new_kwargs_ = fixed_args.copy()
            new_kwargs_.update(new_kwargs)
#             print (new_kwargs_)
            matname_alltrials = make_figname(figname, new_kwargs).replace(figpath, matpath) + '.npy'
            if not(os.path.isfile(matname_alltrials)):
                for i_trial in np.random.permutation(np.arange(N_trials)):
                    figname_this_trial = make_figname(figname, new_kwargs, i_trial)
                    matname_this_trial = figname_this_trial.replace(figpath, matpath) + '.npy'
                    if not(os.path.isfile(matname_this_trial)):
#                         print(matname_this_trial)
#                         print(dict(**fixed_args, new_kwargs))
                        #print(fixed_args.update(dict(new_kwargs, **fixed_args))
                        image_this_trial = generate(N_X, N_Y, N_frame, **new_kwargs_) #dict(new_kwargs, **fixed_args))
                        switch_locked = figure_condensation(figname_this_trial, image_this_trial,
                                                            ext=ext, loops=loops, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                                                            figures=(i_trial==0) and do_figures, video=(i_trial==0) and do_videos,
                                                            **new_kwargs_) #dict(new_kwargs, **fixed_args))
                        if switch_locked == 'lock':
                            switch_finished = False

        if (switch_finished):
            # > gathering all trials in a big vector
            for new_kwargs in kwargs_variable:
                matname_alltrials = make_figname(figname, new_kwargs).replace(figpath, matpath) + '.npy'
#                 print (matname_alltrials)
                if not(os.path.isfile(matname_alltrials)) and not(os.path.isfile(matname_alltrials + '_lock')):
                    touch(matname_alltrials + '_lock')
                    touch(matname_alltrials + LOCK)
                    particles_alltrials = np.zeros((5, N_particles*N_trials, int(N_frame*loops)))
                    for i_trial in range(N_trials):
                        figname_this_trial = make_figname(figname, new_kwargs, i_trial)
                        matname_this_trial = figname_this_trial.replace(figpath, matpath) + '.npy'
                        particles_alltrials[:, (i_trial*N_particles):((i_trial+1)*N_particles), :] = np.load(matname_this_trial)
                    np.save(matname_alltrials, particles_alltrials)
                    try: os.remove(matname_alltrials + LOCK)
                    except: pass
                    try: os.remove(matname_alltrials + '_lock')
                    except: pass
                    for i_trial in range(N_trials):
                        figname_this_trial = make_figname(figname, new_kwargs, i_trial)
                        matname_this_trial = figname_this_trial.replace(figpath, matpath) + '.npy'
                        try: os.remove(matname_this_trial)
                        except: pass
                elif not(os.path.isfile(matname_alltrials + '_lock')):
                    try:
                        particles_alltrials = np.load(matname_alltrials)
                    except:
                        switch_finished = False
                else:
                    print  (matname_alltrials, ' is locked')
                    switch_finished = False
                # > plotting them as one big chunk
                if (do_figure or do_video) and switch_finished:
                    image_this_trial = generate(N_X, N_Y, N_frame, **new_kwargs_) #**dict(new_kwargs, **fixed_args))
                    switch_locked = figure_condensation(make_figname(figname, new_kwargs),
                                                        image_this_trial, particles=particles_alltrials, loops=loops,
                                                        ext=ext, figures=do_figure, video=do_video,
                                                        N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                                                        **new_kwargs_) #**dict(new_kwargs, **fixed_args))
                    if switch_locked == 'lock': switch_finished = False


    if (switch_finished):
#         print(kwargs_variable, list(kwargs_variable[0].keys()))
        kwargs_label = list(kwargs_variable[0].keys())[0]
        figname_CRF = figname + '-' + kwargs_label  + '-CRF' + ext
        figname_OFR = figname + '-' + kwargs_label + '-OFR' + ext
        time = np.linspace(0., T, int(N_frame*loops))
        linestyle = ['*', '+', 'D', 'H', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4']

#         if figures_OFR and (not(os.path.isfile(figname_CRF)) or not(os.path.isfile(figname_OFR))):
        if figures_OFR and not(os.path.isfile(figname_OFR)):
            result = np.zeros((2, N_variable, N_trials, int(N_frame*loops)))
            labels = []
            for i_variable, new_kwargs in enumerate(kwargs_variable):
                matname_alltrials = make_figname(figname, new_kwargs).replace(figpath, matpath) + '.npy'
                p_ = np.load(matname_alltrials)
                for i_trial in range(N_trials):
                    e_h, e_v, v_h, v_v, a_h, a_v = OM(p_[:, (i_trial*N_particles):((i_trial+1)*N_particles), :], display=False)
                    result[0, i_variable, i_trial, :] = e_h
                    result[1, i_variable, i_trial, :] = e_v
                labels.append('%.3f' %  list(new_kwargs.values())[0] )

            # TODO : fix for order==None
            if not(os.path.isfile(figname_CRF)):
                fig, a = PRF(np.arange(N_variable), result[:, :, :, (N_frame//2+latency-5):(N_frame//2+latency+5)].mean(axis=2), kwargs_label, order=order)
                a.set_ylim([-1.5, 1.5])
                a.set_xticks(range(len(labels)))
                a.set_xticklabels(labels)
                fig.savefig(figname_CRF)
                pylab.close(fig)

            if not(os.path.isfile(figname_OFR)):
                fig = pylab.figure(figsize=figsize_PRF)
                a = fig.add_axes((0.15, 0.1, .8, .8))
                step = int(N_frame/N_quant_T)
                lines = a.plot(time[::step], result[0, :, :, ::step].mean(axis=1).T, lw=2, c='r')
                for i_line, line in enumerate(lines):
                    line.set_alpha(float(i_line+1)/len(lines))#
                    line.set_label(labels[i_line])
                lines = a.plot(time[::step], result[1, :, :, ::step].mean(axis=1).T, lw=2, c='b')
                for i_line, line in enumerate(lines):
                    line.set_alpha(float(i_line+1)/len(lines))# [i_line])#set_marker(linestyle[i_line])
                adjust_spines(a, ['left', 'bottom'])
                a.set_xlabel('time (ms)')
                a.set_ylabel('eye velocity')
                a.set_ylim([-1.5, 1.5])
                if make_legend : a.legend(loc='lower left')
                fig.savefig(figname_OFR)
                pylab.close(fig)


    return kwargs_variable

def figure_statespace(name, image, ext=ext, N_blur=N_blur, N_noise=N_noise, N_trials=N_trials, D_x=D_x, D_V=D_V, sigma_noise=sigma_noise, range_blur=range_blur, range_noise=range_noise,X_0_statespace=X_0_statespace, Y_0_statespace=Y_0_statespace, V_X_statespace=V_X_statespace, V_Y_statespace=V_Y_statespace, N_step=N_step, N_show=N_show_statespace, N_show_step=N_show_step, width=width, progress=PROGRESS, mode='noise', figures=True):
    """
    Generates a statespace plot showing how well the condensation algorithms tracks an object of known speed


    """

    # HINT : D_V =-1 means we do condensation without prediction
    v_D_V = np.logspace(-range_blur, range_blur, num=N_blur, base=10)*D_V
    v_D_x = np.logspace(-range_blur, range_blur, num=N_blur, base=10)*D_x
    v_noise = np.logspace(-range_noise, range_noise, num=N_noise, base=10)*noise
#    v_D_V[-1] = -1


    figname_statespace_precision = os.path.join(figpath, name) + '_statespace_precision.pdf'
    figname_statespace_bias = os.path.join(figpath, name) + '_statespace_bias.pdf'
    figname_statespace_precision_init = os.path.join(figpath, name) + '_statespace_precision_init.pdf'
    figname_statespace_bias_init = os.path.join(figpath, name) + '_statespace_bias_init.pdf'
    figname_statespace_phases = os.path.join(figpath, name) + '_statespace_phases.pdf'
    figname_statespace = os.path.join(figpath, name) + '_statespace.pdf'
#     figures = [figname_statespace_precision, figname_statespace_bias, figname_statespace_phases, figname_statespace]
    figures_statespace = 'statespace_figures/'
    mats_statespace = 'statespace_mat/'
    matname_statespace = mats_statespace + name + '_statespace.npy'

    def fname(i_blur, i_noise, i_trial):
        for folder in (mats_statespace, figures_statespace):
            if not(os.path.isdir(folder)):os.mkdir(folder)
        if mode=='noise':
            figname = figures_statespace + name + '_D_x-' + str(i_blur) + '_noise-' + str(i_noise)
        else:
            figname = figures_statespace + name + '_D_V-' + str(i_blur) + '_D_x-' + str(i_noise)

        if i_trial > 0: figname += '_trial-' + str(i_trial)
        matname = figname.replace(figures_statespace, mats_statespace) + '.npy'
        return matname, figname
#
#    for i_noise, noise_ in enumerate(v_noise):
#        for i_trial in range(N_trials):
#            for i_blur, D_x_ in enumerate(v_D_x):
#                matname, figname = fname(i_blur, i_noise, i_trial)
#                try:
#                    if i_trial > 0: os.rename(figname.replace('D_x', 'D_V'), figname)
#                    os.rename(matname.replace('D_x', 'D_V'), matname)
#                except:
#                    print matname.replace('D_x', 'D_V')
        figname = name +'-'+ str(parameter)+'='+ str(value).replace('.', '_')

    def generate_latex_table(N_blur, N_noise, N_step, name, show='all', empty_line=False):
        fig_width = .7/N_blur
        table = '\\begin{tabular}{' + N_blur*'c' + '}%\n'
    #    v_D_V, v_noise = [4, 2, 0], [8, 6, 4, 2, 0] #range(N_blur-1, 0, -3), range(N_blur-1, 0, -2)
    #     v_D_V, v_noise = range(N_blur-1, -1, -1), range(N_blur-1, -1, -1)
        v_D_V, v_noise = range(N_blur-1, -1, -N_step), range(0, N_noise, N_step)

        for i_blur in v_D_V:
            for i_noise in v_noise:
                if (show == 'col' and i_noise == v_noise[len(v_noise)/2]) or  (show == 'row' and i_blur == v_D_V[len(v_D_V)/2]) or (show == 'all'):
                    if mode=='noise':
                        table += '\\includegraphics[width=' + str(fig_width) + '\\textheight]{' + name + '_D_x-' + str(i_blur) + '_noise-' + str(i_noise) + '.png}'
                    else:
                        table += '\\includegraphics[width=' + str(fig_width) + '\\textheight]{' + name + '_D_V-' + str(i_blur) + '_D_x-' + str(i_noise) + '.png}'

                if not(i_noise == v_noise[-1]):
                    table += '&%\n' # to the next cell

            if not(i_blur == v_D_V[-1]):
                table += '\\\\%\n' # to the next row
                if empty_line: table += (N_blur-1)*'&' + '\\\\%\n' # an empty line

        table += '\n\\end{tabular}%\n'
        fic = open(figures_statespace + name + 'table_' + show + '.tex', 'w')
        fic.write(table)
        fic.close()


    # First, do all individual simulations statespace analysis
    switch_break = False
    if not(os.path.isfile(matname_statespace + LOCK)):# and not(os.path.isfile(matname_statespace)):
        # study over 2-3 orders of magnitude
        # main loop
        N_X, N_Y, N_frame = image.shape
        if progress:
            pbar = pyprind.ProgBar(N_blur*N_noise*N_trials, title="State-space")
        shuffle_D_V = np.random.permutation(np.arange(N_blur))
        shuffle_D_x = np.random.permutation(np.arange(N_blur))
        shuffle_noise = np.random.permutation(np.arange(N_noise))
        if mode=='noise':
            for i_noise in range(N_noise):
                image_ = image.copy()
                image_ += v_noise[shuffle_noise[i_noise]] * np.random.randn(N_X, N_Y, N_frame)
                for i_blur in range(N_blur):#enumerate(v_D_x[shuffle_D_x]):
                    for i_trial in range(N_trials):
#                    for i_blur, D_V_ in enumerate(v_D_V[shuffle_D_V]):
                        matname, figname = fname(shuffle_D_x[i_blur], shuffle_noise[i_noise], i_trial)
                        if not(os.path.isfile(matname + LOCK)) and not(os.path.isfile(matname)):
#                            mat_condensation(matname, image_, D_V=D_V_, D_x=v_D_x[shuffle_D_V[i_blur]], progress=False)
                            mat_condensation(matname, image_, D_x=v_D_x[shuffle_D_x[i_blur]], D_V=v_D_V[shuffle_D_x[i_blur]], progress=False)
#                            mat_condensation(matname, image_, D_x=D_x_, progress=False)
#                            else:
#                                # TODO : no prediction = condensation with D_V=0, D_x = inf, resampling = 0
#                                mat_condensation(matname, image_, D_V=0,  progress=False)
                            # if we perform a novel individual run, we should remoe forward dependencies, that is global evaluation of the tracking
                            if os.path.isfile(matname_statespace): os.remove(matname_statespace)
                        if i_trial == 0:
                            if os.path.isfile(matname) and not(os.path.isfile(figname + '.png')):
                                particles = np.load(matname)
                                show_particles(particles[:, :, ::N_show_step], image=image_[:, :, 0]+image_[:, :, -1])
                                pylab.savefig(figname + '.png')
                                pylab.close('all')

                        if progress: pbar.update(i_noise*N_trials*N_blur+i_blur*N_trials+i_trial)
        else:
            for i_D_x in range(N_blur):
                image_ = image.copy()
                image_ += noise * np.random.randn(N_X, N_Y, N_frame)
                for i_trial in range(N_trials):
                    for i_blur in range(N_blur):
                        matname, figname = fname(shuffle_D_x[i_D_x], shuffle_D_V[i_blur], i_trial)
                        if not(os.path.isfile(matname + LOCK)) and not(os.path.isfile(matname)):
                            mat_condensation(matname, image_, D_x=v_D_x[shuffle_D_x[i_blur]], D_V=v_D_V[shuffle_D_V[i_blur]], progress=False)
                            if os.path.isfile(matname_statespace): os.remove(matname_statespace)
                        if i_trial == 0:
                            if os.path.isfile(matname) and not(os.path.isfile(figname + '.png')):
                                particles = np.load(matname)
                                show_particles(particles[:, :, ::N_show_step], image=image_[:, :, 0]+image_[:, :, -1])
                                pylab.savefig(figname + '.png')
                                pylab.close('all')

                        if progress: pbar.update()#i_D_x*N_trials*N_blur+i_trial*N_blur+i_blur)

        generate_latex_table(N_blur, N_noise, N_step, name)
        generate_latex_table(N_blur, N_noise, N_step, name, show='row')
        generate_latex_table(N_blur, N_noise, N_step, name, show='col')


    # routine to evaluate tracking
    def tracker(particles, frame):
        # TODO: computing variability for poisition should use a circular gaussian = von mises
        # TODO: use circular variance of P_i defined over angles \theta_i = 1 - |R| with R = \sum_i P_i e^{i 2 \theta_i} / \sum_i P_i
#                    print particles_.shape, N_frame, frame
        w = particles[4, :, frame]
        w /= w.sum()
        # TODO check the formula for the trajectory... @ frame=-1 we are not yet to the same point as in frame 0!
        particle_phys = [torus((X_0_statespace+frame/float(N_frame)*V_X_statespace*width), width),
                         torus((Y_0_statespace+frame/float(N_frame)*V_Y_statespace*width), width),
                         V_X_statespace, V_Y_statespace]
        diff = particles_[:4, :, frame]-np.array([particle_phys]).T
        tracking = np.sqrt((diff*w).sum(axis=1)**2) # bias = squared of the mean difference
        sigma = np.sqrt((diff**2*w).sum(axis=1)) # precision =  mean squared error
        return np.array([tracking, sigma])


    # Then, evaluate tracking

    if os.path.isfile(matname_statespace):
        tracking = np.load(matname_statespace)
    else:
        if not(os.path.isfile(matname_statespace + LOCK)):
            # check that everything was computed in the first step
            # this is already checked above: we do it again because the individual simulations may have been launched in different runs
            if mode=='noise':
                for i_noise, noise_ in enumerate(v_noise):
                    for i_trial in range(N_trials):
                        for i_blur, D_x_ in enumerate(v_D_x):
                            matname, figname = fname(i_blur, i_noise, i_trial)
        #                     print matname, os.path.isfile(matname + LOCK), not(os.path.isfile(matname))
                            if os.path.isfile(matname + LOCK) or not(os.path.isfile(matname)):
                                switch_break = True
            else:
                for i_D_V, D_V_ in enumerate(v_D_V):
                    for i_trial in range(N_trials):
                        for i_D_x, D_x_ in enumerate(v_D_x):
                            matname, figname = fname(i_D_x, i_D_V, i_trial)
        #                     print matname, os.path.isfile(matname + LOCK), not(os.path.isfile(matname))
                            if os.path.isfile(matname + LOCK) or not(os.path.isfile(matname)):
                                switch_break = True

            # Now, do it
            if not(switch_break) and not(os.path.isfile(matname_statespace + '_lock')):
                # locking
                touch(matname_statespace + LOCK)
                touch(matname_statespace + '_lock')


                # remove forward dependencies, that is figures showing tracking
                for figname in [figname_statespace_precision, figname_statespace_bias, figname_statespace_phases, figname_statespace,
                        figname_statespace_precision_init, figname_statespace_bias_init]:
                    if os.path.isfile(figname): os.remove(figname)

                # tracking : [i_blur, i_noise, bias - precision, x-y-u-v ,  init or last]
                tracking = np.zeros((N_blur, N_noise, 2, 4, 2))#
                if progress:
                    pbar = pyprind.ProgBar(N_blur*N_noise*N_trials, title="State-space")
                if mode=='noise':
                    for i_noise in range(N_noise):
                        for i_trial in range(N_trials):
                            for i_blur in range(N_blur):
                                matname, figname = fname(i_blur, i_noise, i_trial)
                                particles_ = np.load(matname)
                                # doing everything on the last set of particles = particles_[:, :, -1]
                                tracking[i_blur, i_noise, : , :, 0] += tracker(particles_, frame=0)/N_trials
                                tracking[i_blur, i_noise, : , :, 1] += tracker(particles_, frame=N_frame-1)/N_trials

                                if progress: pbar.update()#i_noise*N_trials*N_blur+i_trial*N_blur+i_blur)
                else:
                    for i_D_V, D_V_ in enumerate(v_D_V):

                            for i_trial in range(N_trials):
                                for i_blur in range(N_blur):
                                    matname, figname = fname(i_blur, i_D_V, i_trial)
                                    particles_ = np.load(matname)
                                    # doing everything on the last set of particles = particles_[:, :, -1]
                                    tracking[i_blur, i_D_V, : , :, 0] += tracker(particles_, frame=0)/N_trials
                                    tracking[i_blur, i_D_V, : , :, 1] += tracker(particles_, frame=N_frame-1)/N_trials

                                    if progress: pbar.update()#i_noise*N_trials*N_blur+i_trial*N_blur+i_blur)

                np.save(matname_statespace, tracking)
                os.remove(matname_statespace + LOCK)
                os.remove(matname_statespace + '_lock')
            print('Evaluated tracking')# just did it

        else:
            switch_break = True
            print(matname_statespace, ' locked')# not finished in another process

    v_D_V_text = ['%.3f' % D_V for D_V in v_D_V]
    v_D_x_text = ['%.3f' % D_x for D_x in v_D_x]
    v_noise_text = ['%0.2f' % noise for noise in v_noise]

    def figure_4panels(mat, titles):
        vmax = 0#log10(tracking[: , :, 0, 0].T).max()
        vmin = mat[: , :, 0].min()
        fig = pylab.figure(figsize=(12, 8))
        a1 = fig.add_subplot(221)
        mapable = a1.pcolormesh(mat[: , :, 0], vmin=vmin, vmax=vmax)#, edgecolors='k')
        pylab.axis('tight')
        a2 = fig.add_subplot(222)
        a2.pcolormesh(mat[: , :, 1], vmin=vmin, vmax=vmax)#, edgecolors='k')
        pylab.axis('tight')
        a3 = fig.add_subplot(223)
        a3.pcolormesh(mat[: , :, 2], vmin=vmin, vmax=vmax)#, edgecolors='k')
        pylab.axis('tight')
        a4 = fig.add_subplot(224)
        a4.pcolormesh(mat[: , :, 3], vmin=vmin, vmax=vmax)#, edgecolors='k')
        pylab.axis('tight')

        for i_a, ax in enumerate([a1, a2, a3, a4]):
            ax.set_yticks(np.arange(0, N_noise, 2))
            ax.set_xticks(np.arange(0, N_blur, 2))
            ax.set_title(titles[i_a])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if ax in [a3, a4]:
                ax.set_xticklabels(v_noise_text[::2])
                ax.set_xlabel('external noise')
            if ax in [a1, a3]:
                ax.set_ylabel('internal noise')
                ax.set_yticklabels(v_D_V_text[::2])
            ax.axis('tight')

#        for ax in [a1, a2]:
#            ax.set_xticklabels([])
#            ax.set_xlabel('')
#        for ax in [a2, a4]:
#            ax.set_yticklabels([])
#            ax.set_ylabel('')

        height = pylab.rcParams['figure.subplot.top']-pylab.rcParams['figure.subplot.bottom']
#        print [0.91, pylab.rcParams['figure.subplot.bottom'] + height/4., .025, height/2. ]
        a5 = pylab.axes([0.91, pylab.rcParams['figure.subplot.bottom'] + height/4., .025, height/2. ], axisbg='w') # [l, b, w, h]
        pylab.colorbar(mapable, cax=a5, ax=a1, format='%.2f')
        return fig
#    print tracking.min(), tracking.max(), tracking.mean()
#    print tracking[: , :, 0, :, 0].min(), tracking[: , :, 0, :, 0].max(), tracking[: , :, 0, :, 0].mean()
#    print tracking[: , :, 0, :, 1].min(), tracking[: , :, 0, :, 1].max(), tracking[: , :, 0, :, 1].mean()
#
    # Finally, make figures
    if not(switch_break) and figures:
        # tracking : [i_blur, i_noise, bias - precision, x-y-u-v ,  init or last]
        if not(os.path.isfile(figname_statespace_bias_init)):
                fig = figure_4panels(np.log10(tracking[: , :, 0, :, 0]), ['tracking_X', 'tracking_Y', 'tracking_U', 'tracking_V'])
                fig.savefig(figname_statespace_bias_init)

        if not(os.path.isfile(figname_statespace_precision_init)):
                fig = figure_4panels(np.log10(tracking[: , :, 1, :, 0]), ['sigma_X', 'sigma_Y', 'sigma_U', 'sigma_V'])
                pylab.savefig(figname_statespace_precision_init)

        if not(os.path.isfile(figname_statespace_bias)):
                fig = figure_4panels(np.log10(tracking[: , :, 0, :, 1]), ['tracking_X', 'tracking_Y', 'tracking_U', 'tracking_V'])
                fig.savefig(figname_statespace_bias)

        if not(os.path.isfile(figname_statespace_precision)):
                fig = figure_4panels(np.log10(tracking[: , :, 1, :, 1]), ['sigma_X', 'sigma_Y', 'sigma_U', 'sigma_V'])
                pylab.savefig(figname_statespace_precision)

#
#        if not(os.path.isfile(figname_statespace_phases)):
#                fig = figure_4panels(np.log10(tracking_pos_vel), ['tracking_pos', 'tracking_vel', 'sigma_pos','sigma_vel'])
#                pylab.savefig(figname_statespace_phases)
#
        if not(os.path.isfile(figname_statespace_phases)):
            # tracking : [i_blur, i_noise, bias - precision, x-y-u-v ,  init or last]
#            tracking_pos_vel = np.zeros(tracking[: , :, 0, :, 1].shape) # precision at end
#            tracking_pos_vel[: , :, 0:2] = np.sqrt(.5*(tracking[: , :, :, 0, 1]**2+tracking[: , :, :, 1, 1]**2))
#            tracking_pos_vel[: , :, 2:4] = np.sqrt(.5*(tracking[: , :, :, 2, 1]**2+tracking[: , :, :, 3, 1]**2))
#                # TODO this is a HACK just showing precision
#                mat = np.log10(np.sqrt(tracking[: , :, 1, 0, 1]**2+tracking[: , :, 1, 1, 1]**2))
#                # U,V mat = np.log10(np.sqrt(tracking[: , :, 1, 2, 1]**2+tracking[: , :, 1, 3, 1]**2))
#                vmax = 0#log10(tracking[: , :, 0, 0].T).max()
#                vmin = mat.min()
#                fig = pylab.figure(figsize=(12, 8))
#                ax = fig.add_subplot(111)
#                mapable = ax.pcolormesh(mat, vmin=vmin, vmax=vmax, edgecolors='k')#
##                pylab.contour(v_D_V, v_noise, tracking_pos_vel[: , :, 0].T**2/2.+tracking_pos_vel[: , :, 1].T**2, N=1)
##                pylab.contourf(v_D_V, v_noise, tracking_pos_vel[: , :, 2].T**2/2.+tracking_pos_vel[: , :, 3].T**2, N=1)
##        #             pylab.contour(v_D_V, v_noise, tracking_X[:-1, 1:].T**2/2.+tracking_V[:-1, 1:].T**2, N=1)
##        #             pylab.contourf(v_D_V, v_noise, np.sqrt(sigma_X[:-1, 1:].T**2/4.+sigma_V[:-1, 1:].T**2), N=1)
##                pylab.axis('tight')
##                pylab.text(0.01, 0.012, 'Tracking', fontsize=12, weight='bold')
##                pylab.text(0.01, 0.027, 'No tracking', fontsize=12, weight='bold')
##                pylab.text(0.05, 0.012, 'False\n tracking', fontsize=12, weight='bold')
##                ax.set_xticklabels([])
##                ax.set_yticklabels([])
#                ax.axis('tight')
##                pylab.colorbar(mapable,     ax=ax, format='%.2f')

            tracking_vel = np.zeros(tracking[: , :, 0, 0, 1].shape) # precision at end
            tracking_vel = np.sqrt(.5*(tracking[: , :, 0, 2, 1]**2+tracking[: , :, 0, 3, 1]**2))
            bias_vel = np.zeros(tracking[: , :, 0, 0, 1].shape) # precision at end
            bias_vel = np.sqrt(.5*(tracking[: , :, 1, 2, 1]**2+tracking[: , :, 1, 3, 1]**2))

            fig = pylab.figure(figsize=(5, 10))
            a1 = fig.add_subplot(211)
            mapable = a1.pcolormesh(tracking_vel)#, edgecolors='k')
            pylab.axis('tight')
            a2 = fig.add_subplot(212)
            a2.pcolormesh(bias_vel)#, edgecolors='k')
            pylab.axis('tight')

            for i_a, ax in enumerate([a1, a2]):
#                ax.set_yticks(.5 + np.arange(0, N_noise, 2))
#                ax.set_xticks(.5 + np.arange(0, N_blur, 2))
#                ax.set_title(titles[i_a])
#                ax.set_xticklabels([])
#                ax.set_yticklabels([])
                if ax in [a1]:
                    ax.set_xticklabels(v_noise_text[::2])
                    ax.set_xlabel('external noise')
                if ax in [a1, a2]:
                    ax.set_ylabel('internal noise')
                    ax.set_yticklabels(v_D_V_text[::2])
                ax.axis('tight')

    #        for ax in [a1, a2]:
    #            ax.set_xticklabels([])
    #            ax.set_xlabel('')
    #        for ax in [a2, a4]:
    #            ax.set_yticklabels([])
    #            ax.set_ylabel('')

            height = pylab.rcParams['figure.subplot.top']-pylab.rcParams['figure.subplot.bottom']
    #        print [0.91, pylab.rcParams['figure.subplot.bottom'] + height/4., .025, height/2. ]
            a5 = pylab.axes([0.91, pylab.rcParams['figure.subplot.bottom'] + height/4., .025, height/2. ], axisbg='w') # [l, b, w, h]
            pylab.colorbar(mapable, cax=a5, ax=a1, format='%.2f')
            pylab.savefig(figname_statespace_phases)



def generate_dot(N_X, N_Y, N_frame,
                X_0=X_0, Y_0=Y_0, V_X=V_X, V_Y=V_Y, dot_size=dot_size,
                blank_duration=0., blank_start=0.,
                flash_duration=0., flash_start=0., #flashing=False,
                width=width, im_noise=im_noise, im_contrast=1.,
                NoisyTrajectory=False, traj_Xnoise=0, traj_Vnoise=0, reversal=False,
                hard=False, second_order=False, f=8, texture=False, sf_0=0.15,
                pink_noise=False,
                **kwargs):

    """

    >> pylab.imshow(concatenate((image[16,:,:],image[16,:,:]), axis=-1))

    """
    r_x = width / 2.
    r_y = r_x * N_Y / N_X
    x, y, t = np.mgrid[r_x*(-1+1./(N_X)):r_x*(1-1./(N_X)):1j*N_X,
                    r_y*(-1+1./(N_Y)):r_y*(1-1./(N_Y)):1j*N_Y,
                    0:(1-1./(N_frame+1)):1j*N_frame]

    if NoisyTrajectory:
       V_Y +=  traj_Vnoise/ np.float(N_frame) * np.random.randn(1, N_frame)
       V_X +=  traj_Vnoise/ np.float(N_frame) * np.random.randn(1, N_frame)

    x_ = np.amin([np.abs((x - X_0) - V_X*t*width), width - np.abs((x - X_0) - V_X*t*width)], axis=0)
    y_ = np.amin([np.abs((y - Y_0)- (V_Y )*t*width), width * N_Y / N_X - np.abs((y - Y_0) - V_Y*t*width)], axis=0)

    tube = np.exp(- (x_**2 + y_**2) /2. / dot_size**2) # size is relative to the width of the torus

    if hard : tube = (tube > np.exp(-1./2)) * 1.

    if texture:
        from experiment_cloud import generate as cloud
        texture = cloud(N_X, N_Y, N_frame, V_X=V_X, V_Y=V_Y, sf_0=sf_0, noise=0)
        texture /= np.abs(texture).max()
        tube *= texture
        #np.random.rand(N_X, N_Y, N_frame)

    if second_order:
        from experiment_fullgrating import generate as fullgrating
        tube *= fullgrating(N_X=N_X, N_Y=N_Y, N_frame=N_frame, width=width,
                            V_X=0., noise=0., f=f)

    tube /= tube.max()
    tube *= im_contrast

    # trimming
    if blank_duration>0.:
        N_start = np.floor(blank_start * N_frame)
        N_blank = np.floor(blank_duration * N_frame)
        tube[:, :, N_start:N_start + N_blank] = 0.

    if flash_duration>0.:
        N_start = int(flash_start * N_frame)
        N_flash = int(flash_duration * N_frame)
        # to have stationary flash, set speed to zero
        tube[:, :, :N_start] = 0.
        tube[:, :, (N_start + N_flash):] = 0.

    if reversal:
        # mirroring the second half period
        tube[:, :, N_frame//2:] = tube[::-1, :, N_frame//2:]

    # adding noise
    if pink_noise:
        from MotionClouds import get_grids, envelope_color, random_cloud, rectif
        fx, fy, ft = get_grids(N_X, N_Y, N_frame)
        envelope = envelope_color(fx, fy, ft, alpha=1) #V_X=V_X*N_Y/N_frame, V_Y=V_Y*N_X/N_frame, sf_0=sf_0, B_V=B_V, B_sf=B_sf, B_theta=B_theta)
        noise_image = 2*rectif(random_cloud(envelope))-1
#         print(noise_image.min(), noise_image.max())
        tube += im_noise * noise_image #np.random.randn(N_X, N_Y, N_frame)
    else:
        tube += im_noise * np.random.randn(N_X, N_Y, N_frame)

    return tube
