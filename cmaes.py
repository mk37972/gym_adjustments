#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sept 29 15:36:15 2020

@author: mincheol
"""
import time
import gym
import numpy as np

gym_env = gym.make('NuFingersRotate-v0')

L1 = 0.1
L2 = 0.075


demoData = []
demoData.append(np.load("../NuFingersContinuousDemoLargeFast.npz", allow_pickle=True)) #load the demonstration data from data file
demoData.append(np.load("../NuFingersContinuousDemoLargeSlow.npz", allow_pickle=True)) #load the demonstration data from data file
demoData.append(np.load("../NuFingersContinuousDemoLargeMedium.npz", allow_pickle=True)) #load the demonstration data from data file
demoData.append(np.load("../NuFingersStepDemoLargePositive.npz", allow_pickle=True)) #load the demonstration data from data file
demoData.append(np.load("../NuFingersStepDemoLargeNegative.npz", allow_pickle=True)) #load the demonstration data from data file
demoData.append(np.load("../NuFingersContinuousDemoSmallFast.npz", allow_pickle=True)) #load the demonstration data from data file
demoData.append(np.load("../NuFingersContinuousDemoSmallSlow.npz", allow_pickle=True)) #load the demonstration data from data file
demoData.append(np.load("../NuFingersContinuousDemoSmallMedium.npz", allow_pickle=True)) #load the demonstration data from data file
demoData.append(np.load("../NuFingersStepDemoSmallPositive.npz", allow_pickle=True)) #load the demonstration data from data file
demoData.append(np.load("../NuFingersStepDemoSmallNegative.npz", allow_pickle=True)) #load the demonstration data from data file

pop_size = 1000;
vec_size = 18;
gen_size = 100;
best_size = 100;

population = []
mean = []
variance = []

# Load from the simulation environment
init_mean = np.array([gym_env.env.sim.model.tendon_frictionloss[gym_env.env.sim.model.tendon_name2id('T1_L')],
                      gym_env.env.sim.model.tendon_frictionloss[gym_env.env.sim.model.tendon_name2id('T2_L')],
                      gym_env.env.sim.model.tendon_frictionloss[gym_env.env.sim.model.tendon_name2id('T1_R')],
                      gym_env.env.sim.model.tendon_frictionloss[gym_env.env.sim.model.tendon_name2id('T2_R')],
                      gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('Joint_1_L')],
                      gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('Joint_2_L')],
                      gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('Joint_1_R')],
                      gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('Joint_2_R')],
                      gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('FakeJoint_1_L')],
                      gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('FakeJoint_2_L')],
                      gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_1_L')][1],
                      gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_2_L')][1],
                      gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_1_R')][1],
                      gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_2_R')][1],
                      1e-6,
                      1e-6,
                      1e-6,
                      1e-6,
                      ])

init_mean = np.array([ 7.05016422e+03, 1.15460815e+04, 8.29773002e+03, 1.05545188e+04
, 1.60179110e-01, 1.22531696e+00, 1.95867776e-01, 1.43976498e+00
, 4.05156613e-02, 4.66119940e-02, 4.51858451e-02, 3.29034772e-02
, 2.66864821e-02, 3.49185662e-02, -1.91485966e-03, -1.95716563e-02
, -2.00629219e-02, -3.50941636e-02])
mean.append(init_mean) # from the model
min_values = init_mean * 1e-1
min_values[-8:] = np.array([1e-6, 1e-6, 1e-6, 1e-6, -0.2, -0.2, -0.2, -0.2])
max_values = init_mean * 1e1
max_values[-8:] = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
pop_candidate = mean[0] + (max_values - min_values)/2.0 * np.random.randn(pop_size, vec_size)
pop_after = []
for popul in pop_candidate:
    popul = np.clip(popul, min_values, max_values)
    pop_after.append(popul)
    
population.append(pop_after)
variance.append(np.diag((max_values - min_values)/2.0))

rate_m = 0.5;
rate_c = 0.5;

vel_threshold = 1.0e-3


print("Initial generation\n Parameters: {}".format(mean[0]))

for gen in range(1, gen_size):
    w = []
    for pop in range(pop_size):
        w_comp = 0
        print('Testing Gen Number: {}, Population number:{}'.format(gen-1, pop))
        
        for epsd in range(len(demoData)):
            o, r, d = gym_env.reset()
            optimum = demoData[epsd].get('obs') # the demonstration trajectory
            
            # Set the simulation environment starting states as seen in demonstration
            gym_env.env.sim.model.tendon_frictionloss[gym_env.env.sim.model.tendon_name2id('T1_L')] = population[gen-1][pop][0]
            gym_env.env.sim.model.tendon_frictionloss[gym_env.env.sim.model.tendon_name2id('T2_L')] = population[gen-1][pop][1]
            gym_env.env.sim.model.tendon_frictionloss[gym_env.env.sim.model.tendon_name2id('T1_R')] = population[gen-1][pop][2]
            gym_env.env.sim.model.tendon_frictionloss[gym_env.env.sim.model.tendon_name2id('T2_R')] = population[gen-1][pop][3]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('Joint_1_L')] = population[gen-1][pop][4]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('Joint_2_L')] = population[gen-1][pop][5]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('Joint_1_R')] = population[gen-1][pop][6]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('Joint_2_R')] = population[gen-1][pop][7]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('FakeJoint_1_L')] = population[gen-1][pop][8]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('FakeJoint_1_L2')] = population[gen-1][pop][8]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('FakeJoint_1_R')] = population[gen-1][pop][8]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('FakeJoint_1_R2')] = population[gen-1][pop][8]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('FakeJoint_2_L')] = population[gen-1][pop][9]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('FakeJoint_2_L2')] = population[gen-1][pop][9]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('FakeJoint_2_R')] = population[gen-1][pop][9]
            gym_env.env.sim.model.dof_damping[gym_env.env.sim.model.joint_name2id('FakeJoint_2_R2')] = population[gen-1][pop][9]
            gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_1_L')] = np.array([-population[gen-1][pop][10], population[gen-1][pop][10]])
            gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_1_L2')] = np.array([- population[gen-1][pop][10], population[gen-1][pop][10]])
            gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_2_L')] = np.array([-population[gen-1][pop][11], population[gen-1][pop][11]])
            gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_2_L2')] = np.array([- population[gen-1][pop][11], population[gen-1][pop][11]])
            gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_1_R')] = np.array([-population[gen-1][pop][12], population[gen-1][pop][12]])
            gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_1_R2')] = np.array([- population[gen-1][pop][12], population[gen-1][pop][12]])
            gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_2_R')] = np.array([-population[gen-1][pop][13], population[gen-1][pop][13]])
            gym_env.env.sim.model.jnt_range[gym_env.env.sim.model.joint_name2id('FakeJoint_2_R2')] = np.array([- population[gen-1][pop][13],population[gen-1][pop][13]])
            # xR_init = np.array([-0.0635, 0.127]) + optimum[0]['observation'][0] * np.cos(optimum[0]['observaion'][2])
            # yR_init = np.array([-0.0635, 0.127]) + optimum[0]['observation'][0] * np.sin(optimum[0]['observaion'][2])
            # xL_init = np.array([0.0635, 0.127]) - optimum[0]['observation'][1] * np.cos(optimum[0]['observaion'][3])
            # yL_init = np.array([0.0635, 0.127]) - optimum[0]['observation'][1] * np.sin(optimum[0]['observaion'][3])
            if epsd < 5:
                qR1_init = -34.256960002596415 * np.pi / 180.0 # large 0.12
                qR2_init = -86.59649961596293 * np.pi / 180.0
                qL1_init = 37.98947632145957 * np.pi / 180.0
                qL2_init = -94.62467468881758 * np.pi / 180.0
                
                gym_env.env.sim.data.ctrl[0] = 0.3350334
                gym_env.env.sim.data.ctrl[1] = -0.51339605
                gym_env.env.sim.data.ctrl[2] = -0.33367597
                gym_env.env.sim.data.ctrl[3] = -0.51552609
            else:
                qR1_init = -16.100454896148804 * np.pi / 180.0 # small 0.04
                qR2_init = -82.57223779874832 * np.pi / 180.0
                qL1_init = 15.815771428571429 * np.pi / 180.0
                qL2_init = -82.55785009726873 * np.pi / 180.0
                
                gym_env.env.sim.data.ctrl[0] =  0.14045752
                gym_env.env.sim.data.ctrl[1] = -0.63834073
                gym_env.env.sim.data.ctrl[2] = -0.14038065
                gym_env.env.sim.data.ctrl[3] = -0.63838769
            
            gym_env.env.sim.data.set_joint_qpos('Joint_1_L', population[gen-1][pop][14] + qL1_init)
            gym_env.env.sim.data.set_joint_qpos('Joint_2_L', population[gen-1][pop][15] + qL2_init)
            gym_env.env.sim.data.set_joint_qpos('Joint_1_R', population[gen-1][pop][16] + qR1_init)
            gym_env.env.sim.data.set_joint_qpos('Joint_2_R', population[gen-1][pop][17] + qR2_init)
            for t in range(len(demoData[epsd].get('obs')[0])-1):
                demoact = demoData[epsd].get('acs')[0][t]
                demoact[1] *= 180.0 / np.pi
                demoact[3] *= 180.0 / np.pi
                
                demoobs = optimum[0][t+1]['observation']
                demoobs[2] *= 180.0 / np.pi
                demoobs[3] *= 180.0 / np.pi
                if gym_env.env.sim.data.get_joint_qvel('Joint_1_L') > vel_threshold or gym_env.env.sim.data.get_joint_qvel('Joint_1_L') < -vel_threshold: 
                    # print("1 {}".format(gym_env.env.sim.data.get_joint_qvel('Joint_1_L')))
                    gym_env.env.sim.model.eq_data[0][0] = gym_env.env.sim.data.get_joint_qpos('Joint_1_L')
                if gym_env.env.sim.data.get_joint_qvel('Joint_2_L') > vel_threshold or gym_env.env.sim.data.get_joint_qvel('Joint_2_L') < -vel_threshold: 
                    # print("2 {}".format(gym_env.env.sim.data.get_joint_qvel('Joint_2_L')))
                    gym_env.env.sim.model.eq_data[1][0] = gym_env.env.sim.data.get_joint_qpos('Joint_2_L')
                if gym_env.env.sim.data.get_joint_qvel('Joint_1_R') > vel_threshold or gym_env.env.sim.data.get_joint_qvel('Joint_1_R') < -vel_threshold: 
                    # print("3 {}".format(gym_env.env.sim.data.get_joint_qvel('Joint_1_R')))
                    gym_env.env.sim.model.eq_data[2][0] = gym_env.env.sim.data.get_joint_qpos('Joint_1_R')
                if gym_env.env.sim.data.get_joint_qvel('Joint_2_R') > vel_threshold or gym_env.env.sim.data.get_joint_qvel('Joint_2_R') < -vel_threshold: 
                    # print("4 {}".format(gym_env.env.sim.data.get_joint_qvel('Joint_2_R')))
                    gym_env.env.sim.model.eq_data[3][0] = gym_env.env.sim.data.get_joint_qpos('Joint_2_R')
                
                try: o, r, d, i = gym_env.step(demoact)
                except: 
                    w_comp -= 1.0
                    print("Error occured.. Skipping this step.")
                w_comp += np.exp(-np.linalg.norm(demoobs[:4] - o['observation'][:4]))
                # print(o['observation'][:4])
                # print(demoobs[:4])
                # gym_env.render()
          
        w.append(w_comp)
    
    print('Calculating next generation...')
    temp_w = np.array(w)
    # sorted_w = []
    # ind = []
    ind = temp_w.argsort()
    sorted_w = temp_w[ind]
    # for i in range(best_size):
    #     idx = np.argpartition(temp_w,best_size-1-i)
    #     print(idx)
    #     temp_w = temp_w[idx[:best_size-i]]
    #     sorted_w.append(temp_w[-1])
    #     ind.append(idx[-1])
    
    sorted_w = sorted_w / np.sum(sorted_w)
    
    mean.append(mean[gen-1])
    quality = 0
    for i in range(best_size):
        mean[gen] += rate_m * sorted_w[pop-i] * (population[gen-1][ind[pop-i]] - mean[gen-1])
        quality += w[ind[pop-i]] / best_size / 9190
    
    variance.append((1-rate_c)*variance[gen-1])
    for i in range(best_size):
        variance[gen] += rate_c * sorted_w[pop-i] * (population[gen-1][ind[pop-i]] - mean[gen-1]).reshape(vec_size,1) * (population[gen-1][ind[pop-i]] - mean[gen-1]).reshape(1,vec_size)
    
    pop_candidate = np.random.multivariate_normal(mean[gen],variance[gen],pop_size)
    pop_after = []
    for popul in pop_candidate:
        popul = np.clip(popul, min_values, max_values)
        pop_after.append(popul)
    population.append(pop_after)
    
    if np.linalg.norm(variance[gen]) < 1e-8: 
        print("Finished! Resulting Parameters: {}\n Resulting Covariance: {}".format(mean[gen],variance[gen]))
        break
    else:
        print("Resulting Generation: {}\n Parameters: {}\n Variance: {}\n Variance norm (lower the better): {}\n Quality (closer to 1 the better): {}".format(gen,mean[gen],variance[gen],np.linalg.norm(variance[gen]),quality))
        
        
 # Parameters: [ 5.51035577e+03  8.29767614e+03  5.29048195e+03  8.87507459e+03
 #  1.39216122e+03  1.40917841e+03  1.42059777e+03  1.41101228e+03
 #  1.45727989e-01  1.23904462e+00  1.72142080e-01  1.49730481e+00
 #  1.33023982e+00  1.19468286e+00  1.23339334e+00  1.35221689e+00
 #  1.66537113e-02  2.61672833e-02  1.44998363e-02  2.48770272e-02
 #  4.46986390e-02  5.32000581e-02  4.22922057e-02  3.91230875e-02
 #  8.76870455e-04 -4.64977726e-02  1.90657441e-03 -3.25326982e-02]
 # Variance: [[ 2.25705722e-06  2.71337940e-07  9.06954141e-07  3.00573722e-06
 #  -2.95777326e-08  4.81434839e-09  1.83152953e-08  1.30823070e-08
 #   2.79310410e-11 -2.76989261e-10  5.34481449e-11  1.48929321e-11
 #   1.04123181e-10  3.09175821e-11  1.69544581e-10  1.36754033e-10
 #   3.68358316e-11  1.16926046e-11  8.72788689e-14  1.96365773e-11
 #   3.18862106e-11  3.68652191e-11  5.48376002e-12  3.02281310e-11
 #  -1.54182042e-10 -1.18581108e-10  5.37949856e-11 -1.08264042e-10]
 # [ 2.71337940e-07  8.42011596e-08  1.16412420e-07  3.56894416e-07
 #  -3.07381435e-09  6.76744815e-10  2.05330509e-09  1.61011647e-09
 #   4.34372513e-12 -3.10206782e-11  6.64819573e-12  6.04937354e-12
 #   1.35630331e-11 -2.09003359e-13  2.46486503e-11  1.55592719e-11
 #   4.49739649e-12  1.61836213e-12  8.65913974e-14  2.26924315e-12
 #   4.14550122e-12  5.06048345e-12  7.62547299e-13  3.44136441e-12
 #  -1.81391633e-11 -1.47519028e-11  6.33093649e-12 -1.32267883e-11]
 # [ 9.06954141e-07  1.16412420e-07  3.74587830e-07  1.21396079e-06
 #  -1.18541582e-08  2.04750589e-09  7.56651728e-09  5.11508576e-09
 #   1.11574453e-11 -1.09763148e-10  2.15225970e-11  4.42642803e-12
 #   4.29094913e-11  1.43798449e-11  6.79525088e-11  5.65720399e-11
 #   1.48589874e-11  4.77787213e-12  1.98233891e-14  7.92881385e-12
 #   1.29246602e-11  1.51599028e-11  2.20787156e-12  1.22673349e-11
 #  -6.18669633e-11 -4.80253902e-11  2.18267327e-11 -4.32481093e-11]
 # [ 3.00573722e-06  3.56894416e-07  1.21396079e-06  4.02506657e-06
 #  -3.94426045e-08  6.70027939e-09  2.45873588e-08  1.74470304e-08
 #   3.71716972e-11 -3.69895570e-10  7.08852555e-11  1.91213475e-11
 #   1.38504568e-10  4.20012294e-11  2.22395660e-10  1.81888620e-10
 #   4.90982966e-11  1.55519151e-11  1.36848105e-13  2.62431115e-11
 #   4.28007635e-11  4.93001648e-11  7.41075721e-12  4.07943705e-11
 #  -2.05871792e-10 -1.58428194e-10  7.19044876e-11 -1.44444186e-10]
 # [-2.95777326e-08 -3.07381435e-09 -1.18541582e-08 -3.94426045e-08
 #   4.06140809e-10 -5.92601538e-11 -2.42115121e-10 -1.80088186e-10
 #  -3.47404250e-13  3.73299682e-12 -7.03210173e-13 -1.19418640e-13
 #  -1.33197921e-12 -5.22899477e-13 -2.22708334e-12 -1.69223106e-12
 #  -4.85848926e-13 -1.47362842e-13  1.86371753e-16 -2.57825983e-13
 #  -4.16725295e-13 -4.78058015e-13 -7.09281515e-14 -3.95172899e-13
 #   2.03115448e-12  1.55726761e-12 -7.08906770e-13  1.42743515e-12]
 # [ 4.81434839e-09  6.76744815e-10  2.04750589e-09  6.70027939e-09
 #  -5.92601538e-11  2.80393110e-11  4.70315939e-11  2.29722326e-11
 #   6.59126078e-14 -5.66109511e-13  1.17693434e-13  6.71875138e-14
 #   2.37647500e-13  5.93232183e-14  2.98476823e-13  1.81180485e-13
 #   8.10499745e-14  2.90899635e-14 -1.34592690e-16  4.66769238e-14
 #   7.33583691e-14  8.84121371e-14  2.42571230e-14  8.43192177e-14
 #  -3.48153943e-13 -2.81143678e-13  1.37529069e-13 -2.39487776e-13]
 # [ 1.83152953e-08  2.05330509e-09  7.56651728e-09  2.45873588e-08
 #  -2.42115121e-10  4.70315939e-11  1.61450569e-10  1.01104115e-10
 #   2.18871054e-13 -2.23835454e-12  4.37536421e-13  5.41439655e-14
 #   8.33204456e-13  3.16276959e-13  1.38991411e-12  1.11529519e-12
 #   2.97947116e-13  9.94364309e-14  6.50463875e-16  1.62671105e-13
 #   2.60164252e-13  3.04651740e-13  5.11468037e-14  2.55970147e-13
 #  -1.25488388e-12 -9.81009352e-13  4.51772455e-13 -8.75320319e-13]
 # [ 1.30823070e-08  1.61011647e-09  5.11508576e-09  1.74470304e-08
 #  -1.80088186e-10  2.29722326e-11  1.01104115e-10  9.79406472e-11
 #   1.61393382e-13 -1.82993330e-12  2.98965924e-13  8.65630054e-14
 #   5.70050093e-13  1.77291578e-13  1.04088293e-12  6.55093993e-13
 #   2.13396438e-13  6.15630376e-14  1.00605692e-15  1.12561759e-13
 #   1.86994452e-13  2.06095898e-13  2.72086069e-14  1.72570192e-13
 #  -8.99403690e-13 -6.77575358e-13  3.10722813e-13 -6.53872802e-13]
 # [ 2.79310410e-11  4.34372513e-12  1.11574453e-11  3.71716972e-11
 #  -3.47404250e-13  6.59126078e-14  2.18871054e-13  1.61393382e-13
 #   4.68401294e-16 -3.43145783e-15  6.56988526e-16  2.73665254e-16
 #   1.40760221e-15  1.40840446e-16  2.10593912e-15  1.52226421e-15
 #   4.67053498e-16  1.54446061e-16  5.86859961e-18  2.43354721e-16
 #   4.14952340e-16  4.69471937e-16  9.02703996e-17  3.79856016e-16
 #  -1.94644502e-15 -1.52761280e-15  6.70750800e-16 -1.36466432e-15]
 # [-2.76989261e-10 -3.10206782e-11 -1.09763148e-10 -3.69895570e-10
 #   3.73299682e-12 -5.66109511e-13 -2.23835454e-12 -1.82993330e-12
 #  -3.43145783e-15  3.76027110e-14 -6.34062296e-15 -1.34945367e-15
 #  -1.24110847e-14 -3.82096273e-15 -2.12725273e-14 -1.53706746e-14
 #  -4.53635112e-15 -1.40203854e-15 -3.64652227e-17 -2.41056315e-15
 #  -3.96780533e-15 -4.42014546e-15 -6.43666355e-16 -3.77772488e-15
 #   1.91488427e-14  1.46045289e-14 -6.67133956e-15  1.36568234e-14]
 # [ 5.34481449e-11  6.64819573e-12  2.15225970e-11  7.08852555e-11
 #  -7.03210173e-13  1.17693434e-13  4.37536421e-13  2.98965924e-13
 #   6.56988526e-16 -6.34062296e-15  1.38884484e-15  3.00506853e-16
 #   2.40990943e-15  5.87465573e-16  4.17686278e-15  3.17727067e-15
 #   8.75628551e-16  2.80940531e-16 -3.48012367e-18  4.62931722e-16
 #   7.42698206e-16  8.72557360e-16  1.49174075e-16  7.08016075e-16
 #  -3.64122753e-15 -2.84012019e-15  1.27915536e-15 -2.54075370e-15]
 # [ 1.48929321e-11  6.04937354e-12  4.42642803e-12  1.91213475e-11
 #  -1.19418640e-13  6.71875138e-14  5.41439655e-14  8.65630054e-14
 #   2.73665254e-16 -1.34945367e-15  3.00506853e-16  2.55895098e-15
 #   6.70012898e-16 -5.06963713e-16  6.70939986e-16  3.99108891e-16
 #   2.50155685e-16  1.00338269e-16  1.75338355e-17  1.50336612e-16
 #   1.94757351e-16  2.70421655e-16  7.58050723e-17  2.09582928e-16
 #  -1.19336116e-15 -7.36799935e-16  3.26253108e-16 -8.46497537e-16]
 # [ 1.04123181e-10  1.35630331e-11  4.29094913e-11  1.38504568e-10
 #  -1.33197921e-12  2.37647500e-13  8.33204456e-13  5.70050093e-13
 #   1.40760221e-15 -1.24110847e-14  2.40990943e-15  6.70012898e-16
 #   6.57344291e-15  2.08863038e-15  7.09900520e-15  6.56131827e-15
 #   1.76780254e-15  5.56356056e-16 -3.49362372e-18  9.41194863e-16
 #   1.51053418e-15  1.79665077e-15  2.88415124e-16  1.39391505e-15
 #  -7.30761092e-15 -5.63342915e-15  2.59428121e-15 -4.96585199e-15]
 # [ 3.09175821e-11 -2.09003359e-13  1.43798449e-11  4.20012294e-11
 #  -5.22899477e-13  5.93232183e-14  3.16276959e-13  1.77291578e-13
 #   1.40840446e-16 -3.82096273e-15  5.87465573e-16 -5.06963713e-16
 #   2.08863038e-15  3.49537482e-15  2.01257046e-15  1.91460889e-15
 #   5.24980872e-16  1.23742292e-16 -3.22297965e-17  3.20633679e-16
 #   4.23626555e-16  5.15133854e-16  4.15134071e-17  3.83522217e-16
 #  -2.06082912e-15 -1.53075087e-15  8.62724793e-16 -1.34820490e-15]
 # [ 1.69544581e-10  2.46486503e-11  6.79525088e-11  2.22395660e-10
 #  -2.22708334e-12  2.98476823e-13  1.38991411e-12  1.04088293e-12
 #   2.10593912e-15 -2.12725273e-14  4.17686278e-15  6.70939986e-16
 #   7.09900520e-15  2.01257046e-15  1.64636385e-14  1.03914167e-14
 #   2.72037844e-15  9.21836700e-16  1.99694578e-17  1.44470003e-15
 #   2.39650011e-15  2.73994832e-15  3.95039104e-16  2.10416227e-15
 #  -1.12475288e-14 -8.94717881e-15  4.01906646e-15 -8.18663657e-15]
 # [ 1.36754033e-10  1.55592719e-11  5.65720399e-11  1.81888620e-10
 #  -1.69223106e-12  1.81180485e-13  1.11529519e-12  6.55093993e-13
 #   1.52226421e-15 -1.53706746e-14  3.17727067e-15  3.99108891e-16
 #   6.56131827e-15  1.91460889e-15  1.03914167e-14  1.25547818e-14
 #   2.15845899e-15  7.34278209e-16  1.43717148e-17  1.16110427e-15
 #   1.88349545e-15  2.21250834e-15  1.72969074e-16  1.72823614e-15
 #  -9.03221772e-15 -6.86770397e-15  3.03269683e-15 -6.21405610e-15]
 # [ 3.68358316e-11  4.49739649e-12  1.48589874e-11  4.90982966e-11
 #  -4.85848926e-13  8.10499745e-14  2.97947116e-13  2.13396438e-13
 #   4.67053498e-16 -4.53635112e-15  8.75628551e-16  2.50155685e-16
 #   1.76780254e-15  5.24980872e-16  2.72037844e-15  2.15845899e-15
 #   6.12844037e-16  1.92902326e-16  1.65904886e-18  3.23985713e-16
 #   5.26687479e-16  6.12319916e-16  9.58102903e-17  4.99750218e-16
 #  -2.55189976e-15 -1.96959823e-15  8.89714807e-16 -1.78132606e-15]
 # [ 1.16926046e-11  1.61836213e-12  4.77787213e-12  1.55519151e-11
 #  -1.47362842e-13  2.90899635e-14  9.94364309e-14  6.15630376e-14
 #   1.54446061e-16 -1.40203854e-15  2.80940531e-16  1.00338269e-16
 #   5.56356056e-16  1.23742292e-16  9.21836700e-16  7.34278209e-16
 #   1.92902326e-16  7.07048697e-17  3.36087990e-18  1.04543504e-16
 #   1.68058257e-16  1.99810834e-16  3.71091553e-17  1.62605139e-16
 #  -8.14615719e-16 -6.43498679e-16  2.89110936e-16 -5.68884493e-16]
 # [ 8.72788689e-14  8.65913974e-14  1.98233891e-14  1.36848105e-13
 #   1.86371753e-16 -1.34592690e-16  6.50463875e-16  1.00605692e-15
 #   5.86859961e-18 -3.64652227e-17 -3.48012367e-18  1.75338355e-17
 #  -3.49362372e-18 -3.22297965e-17  1.99694578e-17  1.43717148e-17
 #   1.65904886e-18  3.36087990e-18  3.96184894e-18  1.18509306e-19
 #   5.23674968e-18  3.67918416e-18  1.00094638e-18  3.33850037e-18
 #  -1.36396168e-17 -1.51327337e-17  2.17380664e-18 -1.34652052e-17]
 # [ 1.96365773e-11  2.26924315e-12  7.92881385e-12  2.62431115e-11
 #  -2.57825983e-13  4.66769238e-14  1.62671105e-13  1.12561759e-13
 #   2.43354721e-16 -2.41056315e-15  4.62931722e-16  1.50336612e-16
 #   9.41194863e-16  3.20633679e-16  1.44470003e-15  1.16110427e-15
 #   3.23985713e-16  1.04543504e-16  1.18509306e-19  1.78182079e-16
 #   2.77686992e-16  3.24195781e-16  5.41822387e-17  2.70401812e-16
 #  -1.35974409e-15 -1.04505962e-15  4.80377872e-16 -9.47933314e-16]
 # [ 3.18862106e-11  4.14550122e-12  1.29246602e-11  4.28007635e-11
 #  -4.16725295e-13  7.33583691e-14  2.60164252e-13  1.86994452e-13
 #   4.14952340e-16 -3.96780533e-15  7.42698206e-16  1.94757351e-16
 #   1.51053418e-15  4.23626555e-16  2.39650011e-15  1.88349545e-15
 #   5.26687479e-16  1.68058257e-16  5.23674968e-18  2.77686992e-16
 #   4.75939116e-16  5.36066607e-16  8.31250341e-17  4.39922014e-16
 #  -2.20055097e-15 -1.71640675e-15  7.70625570e-16 -1.55066675e-15]
 # [ 3.68652191e-11  5.06048345e-12  1.51599028e-11  4.93001648e-11
 #  -4.78058015e-13  8.84121371e-14  3.04651740e-13  2.06095898e-13
 #   4.69471937e-16 -4.42014546e-15  8.72557360e-16  2.70421655e-16
 #   1.79665077e-15  5.15133854e-16  2.73994832e-15  2.21250834e-15
 #   6.12319916e-16  1.99810834e-16  3.67918416e-18  3.24195781e-16
 #   5.36066607e-16  6.34483896e-16  1.01029421e-16  5.07400157e-16
 #  -2.54906055e-15 -1.99413738e-15  8.99467847e-16 -1.77768428e-15]
 # [ 5.48376002e-12  7.62547299e-13  2.20787156e-12  7.41075721e-12
 #  -7.09281515e-14  2.42571230e-14  5.11468037e-14  2.72086069e-14
 #   9.02703996e-17 -6.43666355e-16  1.49174075e-16  7.58050723e-17
 #   2.88415124e-16  4.15134071e-17  3.95039104e-16  1.72969074e-16
 #   9.58102903e-17  3.71091553e-17  1.00094638e-18  5.41822387e-17
 #   8.31250341e-17  1.01029421e-16  4.35063540e-17  9.15133337e-17
 #  -4.11533619e-16 -3.43357191e-16  1.58536212e-16 -2.73279140e-16]
 # [ 3.02281310e-11  3.44136441e-12  1.22673349e-11  4.07943705e-11
 #  -3.95172899e-13  8.43192177e-14  2.55970147e-13  1.72570192e-13
 #   3.79856016e-16 -3.77772488e-15  7.08016075e-16  2.09582928e-16
 #   1.39391505e-15  3.83522217e-16  2.10416227e-15  1.72823614e-15
 #   4.99750218e-16  1.62605139e-16  3.33850037e-18  2.70401812e-16
 #   4.39922014e-16  5.07400157e-16  9.15133337e-17  4.47650839e-16
 #  -2.11765520e-15 -1.63873805e-15  7.51500357e-16 -1.47404705e-15]
 # [-1.54182042e-10 -1.81391633e-11 -6.18669633e-11 -2.05871792e-10
 #   2.03115448e-12 -3.48153943e-13 -1.25488388e-12 -8.99403690e-13
 #  -1.94644502e-15  1.91488427e-14 -3.64122753e-15 -1.19336116e-15
 #  -7.30761092e-15 -2.06082912e-15 -1.12475288e-14 -9.03221772e-15
 #  -2.55189976e-15 -8.14615719e-16 -1.36396168e-17 -1.35974409e-15
 #  -2.20055097e-15 -2.54906055e-15 -4.11533619e-16 -2.11765520e-15
 #   1.07303653e-14  8.24193010e-15 -3.73111282e-15  7.49445879e-15]
 # [-1.18581108e-10 -1.47519028e-11 -4.80253902e-11 -1.58428194e-10
 #   1.55726761e-12 -2.81143678e-13 -9.81009352e-13 -6.77575358e-13
 #  -1.52761280e-15  1.46045289e-14 -2.84012019e-15 -7.36799935e-16
 #  -5.63342915e-15 -1.53075087e-15 -8.94717881e-15 -6.86770397e-15
 #  -1.96959823e-15 -6.43498679e-16 -1.51327337e-17 -1.04505962e-15
 #  -1.71640675e-15 -1.99413738e-15 -3.43357191e-16 -1.63873805e-15
 #   8.24193010e-15  6.45046283e-15 -2.90643261e-15  5.75543431e-15]
 # [ 5.37949856e-11  6.33093649e-12  2.18267327e-11  7.19044876e-11
 #  -7.08906770e-13  1.37529069e-13  4.51772455e-13  3.10722813e-13
 #   6.70750800e-16 -6.67133956e-15  1.27915536e-15  3.26253108e-16
 #   2.59428121e-15  8.62724793e-16  4.01906646e-15  3.03269683e-15
 #   8.89714807e-16  2.89110936e-16  2.17380664e-18  4.80377872e-16
 #   7.70625570e-16  8.99467847e-16  1.58536212e-16  7.51500357e-16
 #  -3.73111282e-15 -2.90643261e-15  1.34706676e-15 -2.60612090e-15]
 # [-1.08264042e-10 -1.32267883e-11 -4.32481093e-11 -1.44444186e-10
 #   1.42743515e-12 -2.39487776e-13 -8.75320319e-13 -6.53872802e-13
 #  -1.36466432e-15  1.36568234e-14 -2.54075370e-15 -8.46497537e-16
 #  -4.96585199e-15 -1.34820490e-15 -8.18663657e-15 -6.21405610e-15
 #  -1.78132606e-15 -5.68884493e-16 -1.34652052e-17 -9.47933314e-16
 #  -1.55066675e-15 -1.77768428e-15 -2.73279140e-16 -1.47404705e-15
 #   7.49445879e-15  5.75543431e-15 -2.60612090e-15  5.31114534e-15]]

