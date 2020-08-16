#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:36:15 2020

@author: mincheol
"""
import run

defaultargs = ['--alg=her','--env=FetchPickAndPlaceFragile-v1', '--num_timesteps=0', '--play']

if __name__ == '__main__':
    for dim in [5]:
        for seed in [1000]:
            for pert in ['none','pert','meas','measpert','delay']:
                loadpath = '--load_path=../baselines-master/baselines/models/fpp_demo25bad{}dim_{}'.format(dim,seed)
                filename = '--filename=Force_Distance_data_random_{}_{}{}'.format(dim,pert,seed)
                perturb = '--perturb={}'.format(pert)
                algdim = '--algdim={}'.format(dim)
                
                finalargs = defaultargs + [loadpath, filename, perturb, algdim]
                run.main(finalargs)