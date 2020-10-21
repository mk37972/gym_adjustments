import gym
import numpy as np


"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []
render = 0

def main():
    env = gym.make('FetchPickAndPlaceFragile-v2')
    numItr = 25
    initStateSpace = "random"
    env.reset()
    env.seed(0)
    print("Reset!")
    
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs,numItr)
    
    for i in [3,4,5]:
        fileName = "data_chip_vel"
        fileName += "_" + initStateSpace
        fileName += "_" + str(numItr)
        fileName += "_bad{}dim".format(i)
        fileName += ".npz"
        if i == 4 or i == 5:
            for j in range(numItr):
                for k in range(env._max_episode_steps):
                    observations[j][k]['observation'] = np.concatenate([observations[j][k]['observation'], [0.5]])
                    actions[j][k] = np.concatenate([actions[j][k], [0.0]])
    
        np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file for

def goToGoal(env, lastObs,numItr):

    goal = lastObs['desired_goal'][:3]
    goal[2] += 0.004
    object_rel_pos = lastObs['observation'][3:6]
    grasping_force = 1.0 * env.env.min_grip
    desired_depth = grasping_force / env.env.sim.model.actuator_gainprm[env.env.sim.model.actuator_name2id('robot0:A_WRJ0'), 0]
    desired_depth = 1
#    print(desired_depth)
    desired_pos = 0.025 - desired_depth
    commanding_pos = desired_pos - lastObs['observation'][6]
#    print(env.env.object_fragility)
#    print(commanding_pos)
#    print(env.env.prev_oforce)
#    print("Grasping force:{}, desired depth:{}, desired_pos:{}".format(grasping_force,desired_depth,desired_pos))
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object

    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)
    
    actiondim = 3

    while np.linalg.norm(object_oriented_goal) >= 0.02 and timeStep <= env._max_episode_steps:
        if render: env.render()
        action = []
        for i in range(actiondim): action += [0]
        object_oriented_goal = object_rel_pos.copy()

        action[0] = -(object_oriented_goal[1]+object_oriented_goal[2])*3
        action[1] = (object_oriented_goal[0])*3
        
        action[actiondim-1] = .05 #-(object_oriented_goal[1]+object_oriented_goal[2])*6 #open

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        
        object_rel_pos = obsDataNew['observation'][3:6]
        commanding_pos = desired_pos - obsDataNew['observation'][6]
        if render: print(env.env.prev_force, env.env.prev_oforce, np.linalg.norm(obsDataNew['achieved_goal'][:3]-obsDataNew['desired_goal'][:3]))
        
#    print("Actual Grasping force:{}".format(env.env.sim.data.sensordata[env.env.sim.model.sensor_name2id('l_finger_frc')]+env.env.sim.data.sensordata[env.env.sim.model.sensor_name2id('r_finger_frc')]))

    while np.linalg.norm(obsDataNew['observation'][7:10]) >= 0.01 and timeStep <= env._max_episode_steps :
        if render: env.render()
        action = []
        for i in range(actiondim): action += [0]
        
        goal_rel_pos = obsDataNew['observation'][7:10]
        action[0] = -(goal_rel_pos[1]+goal_rel_pos[2])*3
        action[1] = (goal_rel_pos[0])*3

        action[actiondim-1] = 1

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        object_rel_pos = obsDataNew['observation'][3:6]
        commanding_pos = desired_pos - obsDataNew['observation'][6]
        if render: print(env.env.prev_force, env.env.prev_oforce, np.linalg.norm(obsDataNew['achieved_goal'][:3]-obsDataNew['desired_goal'][:3]))
        
#    print("Actual Grasping force:{}".format(-commanding_pos*env.env.sim.model.actuator_gainprm[env.env.sim.model.actuator_name2id('robot0:l_gripper_finger_joint'), 0]))
    while True: #limit the number of timesteps in the episode to a fixed duration
        if render: env.render()
        action = []
        for i in range(actiondim): action += [0]
        action[actiondim-1] = 1 # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        object_rel_pos = obsDataNew['observation'][3:6]
        if render: print(env.env.prev_force, env.env.prev_oforce, np.linalg.norm(obsDataNew['achieved_goal'][:3]-obsDataNew['desired_goal'][:3]))

        if timeStep >= env._max_episode_steps: break

#    print("Actual Grasping force:{}".format(-commanding_pos*env.env.sim.model.actuator_gainprm[env.env.sim.model.actuator_name2id('robot0:l_gripper_finger_joint'), 0]))
    
    
    if np.linalg.norm(obsDataNew['observation'][7:10]) < env.env.distance_threshold:
        actions.append(episodeAcs)
        observations.append(episodeObs)
        infos.append(episodeInfo)
    else: print("Goal was not reached")


if __name__ == "__main__":
    main()
