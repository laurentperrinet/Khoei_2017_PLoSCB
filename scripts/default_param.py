# -*- coding: utf-8 -*-
"""
Default parameters for all experiments

"""
from __future__ import division, print_function
import numpy as np

import MotionParticles as mp
N_X, N_Y, N_frame = mp.N_X, mp.N_Y, mp.N_frame
X_0 = -1.
V_X = 1.

PBP_D_x = mp.D_x*2.
PBP_D_V = np.inf #mp.D_V*1000.
PBP_prior = mp.v_prior #/1.e6

dot_size = 0.05
dot_start = .2
dot_stop = .8

im_noise_dot = .05
im_noise_flash = .05
#print('Image Noise=', noise)
latencies = [0, 10] # in # of frames
latency = latencies[0]
latency = latencies[-1]
flash_duration = 0.05 # in seconds

stim_labels = [
            'dot',
            'flash',
               ]
stim_args = [
            {'X_0':X_0, 'Y_0':0, 'V_X':V_X, 'im_noise':im_noise_dot, 'hard': True, 'pink_noise': True, 'dot_size':dot_size,
                             'flash_start':dot_start, 'flash_duration':dot_stop-dot_start},
            {'X_0':0., 'Y_0':0., 'V_X':0., 'im_noise':im_noise_flash, 'hard': True, 'pink_noise': True, 'dot_size':dot_size,
                             'flash_duration':flash_duration, 'flash_start':0.5-flash_duration/2},
            ]

# for figures
fontsize = 12
FORMATS = ['.png']
FORMATS = ['.pdf', '.eps', '.png', '.tiff']
FORMATS = ['.pdf']
FORMATS = ['.pdf', '.eps', '.svg']
FORMATS = ['.png', '.pdf']
fig_width_pt = 318.670  # Get this from LaTeX using \showthe\columnwidth
fig_width_pt = 450  # Get this from LaTeX using \showthe\columnwidth
#fig_width_pt = 1024 #221     # Get this from LaTeX using \showthe\columnwidth / x264 asks for a multiple of 2
dpi = 72.27 # dpi settings to get one point per pixel
inches_per_pt = 1.0/dpi            # Convert pt to inches
inches_per_cm = 1./2.54
fig_width = fig_width_pt*inches_per_pt  # width in inches
grid_fig_width = 2*fig_width
phi = (np.sqrt(5) + 1. ) /2
#legend.fontsize = 8
#fig_width = 9
fig_height = fig_width/phi
figsize = (fig_width, fig_height)


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

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

        
import matplotlib
pylab_defaults = { 
    'font.size': 10,
    'xtick.labelsize':'medium',
    'ytick.labelsize':'medium',
    'text.usetex': False,
#    'font.family' : 'sans-serif',
#    'font.sans-serif' : ['Helvetica'],
    }
    
#matplotlib.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
matplotlib.rcParams.update(pylab_defaults)
#matplotlib.rcParams.update({'text.usetex': True})

import matplotlib.cm as cm

# quantization 
N_quant_X = 50
N_quant_Y = 50
N_frame_av = 2 # on how many frames (before AND after) we average
do_video = False
do_figure = True


import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(precision=4)#, suppress=True)

import os