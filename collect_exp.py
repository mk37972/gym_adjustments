#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:36:15 2020

@author: mincheol
"""
import baselines.run as run

# defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v1', '--num_timesteps=0', '--play']

# if __name__ == '__main__':
#     for dim in [6]:
#         for seed in [500]:
#             for pert in ['none','pert','meas','measpert','delay']:
#                 loadpath = '--load_path=./models/block/harsh_65/NoIL/fpp_demo25bad{}dim_{}'.format(dim,seed)
#                 filename = '--filename=./models/block/harsh_65/NoIL/force_dist_data/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
#                 perturb = '--perturb={}'.format(pert)
#                 algdim = '--algdim={}'.format(dim)
                
#                 finalargs = defaultargs + [loadpath, filename, perturb, algdim]
#                 run.main(finalargs)

defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v2', '--num_timesteps=0', '--play']

if __name__ == '__main__':
    for dim in [5]:
        for seed in [500]:
            for pert in ['none','pert','meas','measpert','delay']:
                loadpath = '--load_path=./models/chip/harsh_85/fpp_demo25bad{}dim_{}'.format(dim,seed)
                filename = '--filename=./models/chip/harsh_85/force_dist_data/Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
                perturb = '--perturb={}'.format(pert)
                algdim = '--algdim={}'.format(dim)
                
                finalargs = defaultargs + [loadpath, filename, perturb, algdim]
                run.main(finalargs)