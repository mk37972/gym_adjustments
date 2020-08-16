#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:36:15 2020

@author: mincheol
"""

import run
import mpi4py

defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v1', '--num_timesteps=3e5']
for dim in [5]:
    for seed in [1000]:
        savepath = '--save_path=../baselines-master/baselines/models/fpp_demo25bad{}dim_{}'.format(dim,seed)
        demofile = '--demo_file=data_fetch_random_50_bad{}dim.npz'.format(dim)
        logpath = '--log_path=../baselines-master/baselines/models/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
        perturb = '--perturb=none'
        algdim = '--algdim={}'.format(dim)
#        if seed >= 100 and seed < 1000: seed = 10
#        elif seed >= 1000 and seed < 10000: seed = 100
#        elif seed >= 10000 and seed < 100000: seed = 1000
        
        finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
        run.main(finalargs)
            