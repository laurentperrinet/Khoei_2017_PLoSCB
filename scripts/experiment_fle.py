import MotionParticles as mp
gen_dot = mp.generate_dot
import numpy as np
import os
from default_param import *

image = {}
experiment = 'FLE'
do_sim = False
do_sim = True
for stimulus_tag, im_arg in zip(stim_labels, stim_args):
    # generating the movie
    image[stimulus_tag] = {}
    image[stimulus_tag]['args'] = im_arg
    image[stimulus_tag]['im'] = gen_dot(N_X=N_X, N_Y=N_Y, N_frame=N_frame, **image[stimulus_tag]['args'])
    mp.anim_save(image[stimulus_tag]['im'], os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-anim'))
    image[stimulus_tag]['result'] = {}
    if do_sim:
        # running PX and MBP with 2 different latencies
        for D_x, D_V, v_prior, label in zip([mp.D_x, PBP_D_x], [mp.D_V, PBP_D_V], [mp.v_prior, PBP_prior], ['MBP', 'PBP']):
            figname = os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label)
            
            image[stimulus_tag]['result'][label] = {}
            image[stimulus_tag]['args'].update(D_V=D_V, D_x=D_x, v_prior=v_prior)
            _  = mp.figure_image_variable(
                    figname, 
                    N_X, N_Y, N_frame, gen_dot, order=None, 
                    do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                    fixed_args=image[stimulus_tag]['args'], latency=latencies)
            try:
                for latency in latencies:
                    matname = mp.make_figname(figname, {'latency': latency}).replace(mp.figpath, mp.matpath) + '.npy'
                    image[stimulus_tag]['result'][label][latency] = np.load(matname)
            except:
                print('no result yet for ', matname)