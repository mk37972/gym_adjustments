#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:36:15 2020

@author: mincheol
"""
import baselines.run as run

defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v1', '--num_timesteps=0', '--play']

if __name__ == '__main__':
    for dim in [6]:
        for seed in [10,100,1000]:
            for pert in ['none','pert','meas','measpert','delay']:
                loadpath = '--load_path=./models/yesbuffer/fpp_demo25bad{}dim_{}'.format(dim,seed)
                filename = '--filename=./models/yesbuffer/force_dist_data/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
                perturb = '--perturb={}'.format(pert)
                algdim = '--algdim={}'.format(dim)
                
                finalargs = defaultargs + [loadpath, filename, perturb, algdim]
                run.main(finalargs)

# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v2', '--num_timesteps=0', '--play']

# if __name__ == '__main__':
#     for dim in [3]:
#         for seed in [10,100,1000]:
#             for pert in ['none','pert','meas','measpert','delay']:
#                 loadpath = '--load_path=./models/chip/fpp_demo25bad{}dim_{}'.format(dim,seed)
#                 filename = '--filename=./models/chip/force_dist_data/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
#                 perturb = '--perturb={}'.format(pert)
#                 algdim = '--algdim={}'.format(dim)
                
#                 finalargs = defaultargs + [loadpath, filename, perturb, algdim]
#                 run.main(finalargs)