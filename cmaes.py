#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sept 29 15:36:15 2020

@author: mincheol
"""

import gym
import numpy as np

gym_env = gym.make('SimulationEnv')

demoData1 = np.load("NuFingersContinuousDemoLargeFast.npz", allow_pickle=True) #load the demonstration data from data file
demoData2 = np.load("NuFingersContinuousDemoLargeSlow.npz", allow_pickle=True) #load the demonstration data from data file
demoData3 = np.load("NuFingersContinuousDemoLargeMedium.npz", allow_pickle=True) #load the demonstration data from data file
demoData4 = np.load("NuFingersContinuousDemoSmallFast.npz", allow_pickle=True) #load the demonstration data from data file
demoData5 = np.load("NuFingersContinuousDemoSmallSlow.npz", allow_pickle=True) #load the demonstration data from data file
demoData6 = np.load("NuFingersContinuousDemoSmallMedium.npz", allow_pickle=True) #load the demonstration data from data file
demoData7 = np.load("NuFingersStepDemoPlus.npz", allow_pickle=True) #load the demonstration data from data file
demoData8 = np.load("NuFingersStepDemoMinus.npz", allow_pickle=True) #load the demonstration data from data file
demoData9 = np.load("NuFingersStepDemoZero.npz", allow_pickle=True) #load the demonstration data from data file

demo_data_obs = np.concatenate([demoData1['obs'],
                                demoData2['obs'],
                                demoData3['obs'],
                                demoData4['obs'],
                                demoData5['obs'],
                                demoData6['obs'],
                                demoData7['obs'],
                                demoData8['obs'],
                                demoData9['obs']])
demo_data_acs = np.concatenate([demoData1['acs'],
                                demoData2['acs'],
                                demoData3['acs'],
                                demoData4['acs'],
                                demoData5['acs'],
                                demoData6['acs'],
                                demoData7['acs'],
                                demoData8['acs'],
                                demoData9['acs']])
demo_data_info = np.concatenate([demoData1['info'],
                                demoData2['info'],
                                demoData3['info'],
                                demoData4['info'],
                                demoData5['info'],
                                demoData6['info'],
                                demoData7['info'],
                                demoData8['info'],
                                demoData9['info']])

pop_size = 1000;
vec_size = 55;
gen_size = 1000;
best_size = 100;

population = []
mean = []
variance = []

# Load from the simulation environment
init_mean = np.array([])
mean.append(init_mean) # from the model
population.append(mean[0] + 0.2 * mean[0] * np.random.randn(pop_size, vec_size))
variance.append(0.2 * mean[0] * np.identity(vec_size))

rate_m = 0.5;
rate_c = 0.5;

for gen in range(1, gen_size):
    w = []
    for pop in range(pop_size):
        w_comp = 0
        SetEnv(population[pop]) # Set the simulation environment starting states as seen in demonstration
        for epsd in range(len(demo_data_obs)):
            o, r, d, i = gym_env.reset()
            optimum = demo_data_obs[epsd] # the demonstration trajectory
            for t in range(gym_env.spec.max_episode_steps-1):
                o, r, d, i = gym_env.step(demo_data_acs[epsd][t])
                w_comp += np.exp(-np.linalg.norm(optimum[t+1]['observation'] - o['observation']))
          
        w.append(w_comp)
        
    temp_w = w
    sorted_w = []
    ind = []
    for i in range(best_size):
        idx = np.argpartition(temp_w,best_size-1-i)
        temp_w = temp_w[idx[:best_size-i]]
        sorted_w.append(temp_w[-1])
        ind.append(idx[-1])
    
    sorted_w = sorted_w / np.sum(sorted_w);
    
    mean[gen] = mean[gen-1];
    for i in range(best_size):
        mean[gen] += rate_m * sorted_w[i] * (population[gen-1][ind[i]] - mean[gen-1]);
    
    variance[gen] = (1-rate_c)*variance[gen-1];
    for i in range(best_size):
        variance[gen] = variance[gen] + rate_c * sorted_w[i] * np.matmul((population[gen-1][ind[i]] - mean[gen-1]),np.transpose((population[gen-1][ind[i]] - mean[gen-1])))
    
    population.append(np.random.multivariate_normal(mean[gen],variance[gen],pop_size))
    
    if np.norm(variance[gen]) < 1e-8: 
        print("Resulting Parameters: {}\
              Resulting Covariance: {}".format(mean[gen],variance[gen]))
        break
    else:
        print("Generation: {}\
              Parameters: {}\
                  Quality: {}".format(gen,mean[gen],np.norm(w)))

