
import MotionParticlesFLE as mp
gen_dot = mp.generate_dot
import numpy as np
import os
from default_param import *

image_speed = {}
experiment = 'speed'

speeds = np.linspace(3.75, 1., 11, endpoint=True)
# make such that the dot crosses the middle of the screen at t=.5 while making the same distance
X_0s = -speeds
dot_starts = .5 - .4 / speeds
flash_durations = .8 / speeds

stimulus_tag = stim_labels[0] # 'dot'
im_arg = stim_args[0]

# generating the movie
image_speed[stimulus_tag] = {}
image_speed[stimulus_tag]['args'] = {'Y_0':0,  'im_noise':mp.im_noise,  'dot_size':dot_size}
image_speed[stimulus_tag]['im'] = gen_dot(N_X=N_X, N_Y=N_Y, N_frame=N_frame, **image_speed[stimulus_tag]['args'])

image_speed[stimulus_tag]['result'] = {}
# running PX and MBP with 2 different latencies
for D_x, D_V, v_prior, label in zip([mp.D_x, PBP_D_x], [mp.D_V, PBP_D_V], [mp.v_prior, PBP_prior], ['MBP', 'PBP']):
    figname = os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label)

    image_speed[stimulus_tag]['result'][label] = {}
    image_speed[stimulus_tag]['args'].update(D_V=D_V, D_x=D_x, v_prior=v_prior)

    kwargs_variable  = mp.figure_image_variable(
            figname, 
            N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
            fixed_args=image_speed[stimulus_tag]['args'],
            V_X=speeds, X_0=X_0s, flash_start=dot_starts, flash_duration=flash_durations)

    for new_kwargs in kwargs_variable:
        try:
            matname = mp.make_figname(figname, new_kwargs).replace(mp.figpath, mp.matpath) + '.npy'
            image_speed[stimulus_tag]['result'][label][new_kwargs['V_X']] = np.load(matname)
        except:
            print('no result yet for ', matname)