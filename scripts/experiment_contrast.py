import MotionParticles as mp
gen_dot = mp.generate_dot
import numpy as np
import matplotlib.pyplot as plt
import os
from default_param import *

image_contrast = {}
experiment = 'contrast'

if True:
    im_contrasts = mp.im_contrast*np.logspace(-.6, .125, 9, endpoint=True, base=16)[1:-1]
    for stimulus_tag, im_arg in zip(stim_labels, stim_args):
        # generating the movie
        image_contrast[stimulus_tag] = {}
        image_contrast[stimulus_tag]['args'] = im_arg
        #image_contrast[stimulus_tag]['args'] = {'Y_0':0,  'noise':noise,  'dot_size':dot_size}
        image_contrast[stimulus_tag]['im'] = gen_dot(N_X=N_X, N_Y=N_Y, N_frame=N_frame, **image_contrast[stimulus_tag]['args'])
        image_contrast[stimulus_tag]['result'] = {}
        # running PX and MBP 
        #for D_x, D_V, v_prior, label in zip([mp.D_x, PBP_D_x], [mp.D_V, PBP_D_V], [mp.v_prior, PBP_prior], ['MBP', 'PBP']):
        for D_x, D_V, v_prior, label in zip([mp.D_x], [mp.D_V], [mp.v_prior], ['MBP']):
            figname = os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label)
            #print(figname)

            image_contrast[stimulus_tag]['result'][label] = {}
            image_contrast[stimulus_tag]['args'].update(D_V=D_V, D_x=D_x, v_prior=v_prior)
            kwargs_variable  = mp.figure_image_variable(
                    figname, 
                    N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, 
                    N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                    fixed_args=image_contrast[stimulus_tag]['args'], 
                    im_contrast=im_contrasts)

            for new_kwargs in kwargs_variable:
                if True: #try:
                    matname = mp.make_figname(figname, new_kwargs).replace(mp.figpath, mp.matpath) + '.npy'
                    particles = np.load(matname)
                    image_contrast[stimulus_tag]['result'][label][new_kwargs['im_contrast']] = particles
                    #print(particles.shape)
                    print('>>> Stimulus ', stimulus_tag, label, ' at contrast= ', new_kwargs['im_contrast'] )
                    fig, axs = mp.spatial_readout(particles, N_quant_X=N_quant_Y, N_quant_Y=1)#, fig=fig, a=axs[i])
                    plt.show()

                #except:
                #    print('no result yet for ', matname)

if True:                
    experiment = 'duration'
    flash_durations = np.array([.03, .05, .08, .13, .25])
    flash_starts = .5 - flash_durations/2
    #print(stim_labels[1], stim_args[1], flash_durations, flash_starts)
    #for stimulus_tag, im_arg in zip(stim_labels[1], stim_args[1]):
    stimulus_tag, im_arg = stim_labels[1], stim_args[1]
    image_duration = {}
    image_duration[stimulus_tag] = {}
    image_duration[stimulus_tag]['args'] = im_arg
    #print(stimulus_tag, im_arg)
    image_duration[stimulus_tag]['result'] = {}
    # running PX and MBP with 2 different latencies
    #for D_x, D_V, v_prior, label in zip([mp.D_x, PBP_D_x], [mp.D_V, PBP_D_V], [mp.v_prior, PBP_prior], ['MBP', 'PBP']):
    label = 'MBP'
    figname = os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label)
    image_duration[stimulus_tag]['result'][label] = {}
    kwargs_variable  = mp.figure_image_variable(
            figname, 
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
            fixed_args=image_duration[stimulus_tag]['args'], 
            flash_start=flash_starts, flash_duration=flash_durations)

    for new_kwargs in kwargs_variable:
        try:
            matname = mp.make_figname(figname, new_kwargs).replace(mp.figpath, mp.matpath) + '.npy'
            particles = np.load(matname)
            image_duration[stimulus_tag]['result'][label][new_kwargs['flash_duration']] = particles
            #print('>>> Stimulus ', stimulus_tag, label, ' at flash duration= ', new_kwargs['flash_duration'] )
            #fig, axs = mp.spatial_readout(particles, N_quant_X=N_quant_Y, N_quant_Y=1)#, fig=fig, a=axs[i])
            #plt.show()
        except:
            print('no result yet for ', matname)
    