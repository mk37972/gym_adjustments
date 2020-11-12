#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:36:15 2020

@author: mincheol
"""

from baselines import run
import mpi4py

# defaultargs = ['--alg=her','--env=relocate-v0', '--num_timesteps=0e5', '--play']
# for dim in [6]:
#     for seed in [1000]:
#         # loadpath = '--load_path=./hand_dapg/dapg/policies/relocate.pb'
#         savepath = '--save_path=./models/relocate/fpp_demo25bad{}dim_{}'.format(dim,seed)
#         # demofile = '--demo_file=./hand_dapg/dapg/utils/demo_data.npz'
#         logpath = '--log_path=./models/relocate/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=none'
#         algdim = '--algdim={}'.format(dim)
# #        if seed >= 100 and seed < 1000: seed = 10
# #        elif seed >= 1000 and seed < 10000: seed = 100
# #        elif seed >= 10000 and seed < 100000: seed = 1000
        
#         finalargs = defaultargs + [savepath, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)
     
defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v2', '--num_timesteps=3e5']
for dim in [5]:
    for seed in [10,100,1000]:
        savepath = '--save_path=./models/chip/vel_smooth/harsh_07/IR/fpp_demo25bad{}dim_{}'.format(dim,seed)
        demofile = '--demo_file=./gym_adjustments/data_chip_vel_random_25_bad{}dim.npz'.format(dim)
        logpath = '--log_path=./models/chip/vel_smooth/harsh_07/IR/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
        perturb = '--perturb=delay'
        algdim = '--algdim={}'.format(dim)
#        if seed >= 100 and seed < 1000: seed = 10
#        elif seed >= 1000 and seed < 10000: seed = 100
#        elif seed >= 10000 and seed < 100000: seed = 1000
        
        finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
        run.main(finalargs)
     
# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v1', '--num_timesteps=2e5']
# for dim in [4]:
#     for seed in [100]:
#         savepath = '--save_path=./models/block/harsh_06/fpp_demo25bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./gym_adjustments/data_fetch_random_25_bad{}dim.npz'.format(dim)
#         logpath = '--log_path=./models/block/harsh_06/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
# #        if seed >= 100 and seed < 1000: seed = 10
# #        elif seed >= 1000 and seed < 10000: seed = 100
# #        elif seed >= 10000 and seed < 100000: seed = 1000
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)



# defaultargs = ['--alg=her','--env=NuFingers', '--num_timesteps=1e5']
# for dim in [6]:
#     for seed in [10,100,1000]:
#         savepath = '--save_path=./models/NuFingers/fpp_demo25bad{}dim_{}'.format(dim,seed)
#         demofile = '--demo_file=./NuFingersObjectDemo_bad{}dim.npz'.format(dim)
#         logpath = '--log_path=./models/NuFingers/fpp_demo25bad{}dim_{}_log'.format(dim,seed)
#         perturb = '--perturb=none'
#         algdim = '--algdim={}'.format(dim)
# #        if seed >= 100 and seed < 1000: seed = 10
# #        elif seed >= 1000 and seed < 10000: seed = 100
# #        elif seed >= 10000 and seed < 100000: seed = 1000
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)