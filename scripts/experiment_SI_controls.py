"""
A bunch of control runs

"""
import MotionParticlesFLE as mp
gen_dot = mp.generate_dot
import numpy as np
import os
from default_param import *

image = {}
experiment = 'SI'
N_scan = 5
base = 10.

#mp.N_trials = 4
for stimulus_tag, im_arg in zip(stim_labels, stim_args):
#for stimulus_tag, im_arg in zip(stim_labels[1], stim_args[1]):
    #for D_x, D_V, label in zip([mp.D_x, PBP_D_x], [mp.D_V, PBP_D_V], ['MBP', 'PBP']):
    for D_x, D_V, label in zip([mp.D_x], [mp.D_V], ['MBP']):
        im_arg.update(D_V=D_V, D_x=D_x)

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                D_x=im_arg['D_x']*np.logspace(-2, 2, N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                D_V=im_arg['D_V']*np.logspace(-2, 2, N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                sigma_motion=mp.sigma_motion*np.logspace(-1., 1., N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                K_motion=mp.K_motion*np.logspace(-1., 1., N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                dot_size=im_arg['dot_size']*np.logspace(-1., 1., N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                sigma_I=mp.sigma_I*np.logspace(-1, 1, N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                im_noise=mp.im_noise*np.logspace(-1, 1, N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                sigma_noise=mp.sigma_noise*np.logspace(-1, 1, N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                p_epsilon=mp.p_epsilon*np.logspace(-1, 1, N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                v_init=mp.v_init*np.logspace(-1., 1., N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                v_prior=np.logspace(-.3, 5., N_scan, base=base))

        _  = mp.figure_image_variable(
                os.path.join(mp.figpath, experiment + '-' + stimulus_tag + '-' + label),
                N_X, N_Y, N_frame, gen_dot, order=None, do_figure=do_figure, do_video=do_video, N_quant_X=N_quant_X, N_quant_Y=N_quant_Y,
                fixed_args=im_arg,
                resample=np.linspace(0.1, 1., N_scan, endpoint=True))
