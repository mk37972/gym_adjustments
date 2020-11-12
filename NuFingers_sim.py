# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:01:56 2020

@author: mk37972
"""

import gym
import numpy as np
a = gym.make('NuFingersRotate-v0')

a.reset()

for i in range(1000):
    a.step(np.concatenate([(np.random.random(4)*2-1), [0.0, 0.0]]))
    a.render()
    