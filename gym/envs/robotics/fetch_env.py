import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, fragile_on=False, stiffness_on=False, series_on=False, parallel_on=False, pert_type='none', n_actions=4
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.model_path = model_path
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = 0.05
        self.reward_type = reward_type
        self.fragile_on = fragile_on
        self.stiffness_on = stiffness_on
        self.series_on = series_on
        self.parallel_on = parallel_on
        self.broken_table = False
        self.broken_object = False
        self.max_stiffness = 250.0 if self.model_path.find('wall') == -1 else 50.0
        self.prev_stiffness = self.max_stiffness
        self.psv_prev_stiffness = self.max_stiffness
        self.par_prev_stiffness = 0.0
        self.object_fragility = 0.0
        self.min_grip = 0.0
        self.fric_mu = 0.2
        self.grav_const = 9.81
        self.prev_force = 0.0
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_oforce = 0.0
        self.n_actions = n_actions
        self.previous_input = 0
        self.pert_type = pert_type
        self.remaining_timestep = 50
        self.goaldim = 3 if self.model_path.find('wall') == -1 else 6

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        if self.fragile_on:
            try: 
                d = goal_distance(achieved_goal[:,:self.goaldim], goal[:,:self.goaldim])
                fragile_goal = np.linalg.norm((achieved_goal[:,self.goaldim:] - goal[:,self.goaldim:])*((achieved_goal[:,self.goaldim:] - goal[:,self.goaldim:]) < 0), axis=-1) if self.model_path.find('wall') == -1 else np.linalg.norm((achieved_goal[:,self.goaldim:] - goal[:,self.goaldim:])*((achieved_goal[:,self.goaldim:] - goal[:,self.goaldim:]) > 0), axis=-1)
            except: 
                d = goal_distance(achieved_goal[:self.goaldim], goal[:self.goaldim])
                fragile_goal = np.linalg.norm((achieved_goal[self.goaldim:] - goal[self.goaldim:])*((achieved_goal[self.goaldim:] - goal[self.goaldim:]) < 0), axis=-1) if self.model_path.find('wall') == -1 else np.linalg.norm((achieved_goal[self.goaldim:] - goal[self.goaldim:])*((achieved_goal[self.goaldim:] - goal[self.goaldim:]) > 0), axis=-1)
            if self.reward_type == 'sparse':
                # print(d)
                return -(d > self.distance_threshold).astype(np.float32) - np.float32(fragile_goal) * 2.# - np.float32(penalty)/50.0
            else:
                return -d
        else:
            d = goal_distance(achieved_goal, goal)
            if self.reward_type == 'sparse':
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -d
        

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope
        if self.model_path.find('wall') == -1:
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            stiffness_ctrl = 0.0
            psv_stiffness_ctrl = 0.0
            par_stiffness_ctrl = 0.0
    
            pos_ctrl *= 0.05  # limit maximum change in position
            rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
            
            if action.shape[0] > 4:
                if action.shape[0] > 5:
                    if action.shape[0] > 6:
                        par_stiffness_ctrl = 10.0 * action[6]
                        
                        self.par_prev_stiffness += par_stiffness_ctrl
                        self.par_prev_stiffness = np.max([np.min([self.par_prev_stiffness, 50.0]), 0.0])
    #                    self.psv_prev_stiffness -= self.par_prev_stiffness
                        
                        self.sim.model.jnt_stiffness[self.sim.model.joint_name2id('robot0:r_gripper_finger_joint')] = self.par_prev_stiffness
                        self.sim.model.jnt_stiffness[self.sim.model.joint_name2id('robot0:l_gripper_finger_joint')] = self.par_prev_stiffness
                    
                    psv_stiffness_ctrl = 0.2 * self.max_stiffness * action[5]
                    
                    self.psv_prev_stiffness += psv_stiffness_ctrl
                    self.psv_prev_stiffness = np.max([np.min([self.psv_prev_stiffness, self.max_stiffness - self.par_prev_stiffness]), self.max_stiffness / 25.0])
                
    #                self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('robot0:l_gripper_finger_joint'), 0] = self.psv_prev_stiffness
    #                self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('robot0:r_gripper_finger_joint'), 0] = self.psv_prev_stiffness
    #                self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('robot0:l_gripper_finger_joint'), 1] = -self.psv_prev_stiffness
    #                self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('robot0:r_gripper_finger_joint'), 1] = -self.psv_prev_stiffness
            
                stiffness_ctrl = 0.2 * self.max_stiffness * action[4]
                
                self.prev_stiffness += stiffness_ctrl
                self.prev_stiffness = np.max([np.min([self.prev_stiffness, self.par_prev_stiffness + self.psv_prev_stiffness]), self.par_prev_stiffness])
            
    #        print("Series:{}, Parallel:{}, Active:{}".format(self.psv_prev_stiffness,self.par_prev_stiffness, self.prev_stiffness))
            
            gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
            assert gripper_ctrl.shape == (2,)
            if self.block_gripper:
                gripper_ctrl = np.zeros_like(gripper_ctrl)
            prob = 0.1 if self.pert_type == 'delay' else -0.1
            if np.random.random() > prob:
                action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl, [self.prev_stiffness]])
                self.previous_input = action[7:]
            else:
                try: action = np.concatenate([pos_ctrl, rot_ctrl, self.previous_input])
                except: action = np.concatenate([pos_ctrl, rot_ctrl, np.zeros(3)])
    #        print("Gripper pose:{}, stiffness:{}".format(gripper_ctrl, self.prev_stiffness))
    #        self.prev_stiffness = self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('robot0:l_gripper_finger_joint'), 0]
        else:
            pos_ctrl, gripper_ctrl = action[:2], action[2]
            stiffness_ctrl = 0.0
            psv_stiffness_ctrl = 0.0
    
            pos_ctrl *= 0.05  # limit maximum change in position
            
            if action.shape[0] > 3:
                if action.shape[0] > 4:
                    psv_stiffness_ctrl = 0.2*self.max_stiffness * action[4]
                    
                    self.psv_prev_stiffness += psv_stiffness_ctrl
                    self.psv_prev_stiffness = np.max([np.min([self.psv_prev_stiffness, self.max_stiffness - self.par_prev_stiffness]), 10.0])
                
                stiffness_ctrl = 0.2*self.max_stiffness * action[3]
                
                self.prev_stiffness += stiffness_ctrl
                self.prev_stiffness = np.max([np.min([self.prev_stiffness, self.par_prev_stiffness + self.psv_prev_stiffness]), self.par_prev_stiffness])
            
            prob = 0.1 if self.pert_type == 'delay' else -0.1
            if np.random.random() > prob:
                action = np.concatenate([pos_ctrl, [gripper_ctrl], [self.prev_stiffness]])
                self.previous_input = action[2:]
            else:
                try: action = np.concatenate([pos_ctrl, self.previous_input])
                except: action = np.concatenate([pos_ctrl, np.zeros(2)])
            
            
        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action, self.stiffness_on, Chip=(self.model_path.find('wall') != -1))
        if self.model_path.find('wall') == -1: utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        self.remaining_timestep -= 1
#        print(self.remaining_timestep)
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            if self.pert_type == 'none' or self.pert_type == 'pert':
                object_pos = self.sim.data.get_body_xpos('object0')
            else:
                object_pos = self.sim.data.get_body_xpos('object0') + 0.02 * (np.random.random(3) - 0.5)
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            grip_rel_pos = self.goal.copy()[:3] - grip_pos
            if self.pert_type != 'none' and self.pert_type != 'meas':
                ind = 1 if self.model_path.find('wall') == -1 else 0
                if np.linalg.norm(object_rel_pos) < 0.03: self.sim.data.qvel[self.sim.model.joint_name2id('object0:joint')+ind] += np.random.random()*1.0-0.5
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        
        goal_rel_pos = self.goal.copy()[:3] - object_pos

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        
        if self.fragile_on:
#            if self.sim.data.sensordata[2] > 100 and self.broken_table == False:
#                self.sim.model.geom_matid[self.sim.model.body_geomadr[self.sim.model.body_name2id('table0')]] = 2
#                self.broken_table = True
            if self.model_path.find('wall') == -1:
                l_finger_force = (self.sim.data.ctrl[0] - self.sim.data.sensordata[self.sim.model.sensor_name2id('l_finger_jnt')]) * self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('robot0:l_gripper_finger_joint'), 0] - self.sim.data.sensordata[self.sim.model.sensor_name2id('l_finger_jnt')] * self.sim.model.jnt_stiffness[self.sim.model.joint_name2id('robot0:l_gripper_finger_joint')]
                r_finger_force = (self.sim.data.ctrl[1] - self.sim.data.sensordata[self.sim.model.sensor_name2id('r_finger_jnt')]) * self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('robot0:r_gripper_finger_joint'), 0] - self.sim.data.sensordata[self.sim.model.sensor_name2id('r_finger_jnt')] * self.sim.model.jnt_stiffness[self.sim.model.joint_name2id('robot0:r_gripper_finger_joint')]
                
                l_finger_force = self.prev_lforce + (l_finger_force - self.prev_lforce) * dt / 0.05
                r_finger_force = self.prev_rforce + (r_finger_force - self.prev_rforce) * dt / 0.05
                
                object_force = self.prev_oforce + (self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')] - self.prev_oforce) * dt / 0.1
                
                if (object_force > self.object_fragility):
                    self.sim.model.geom_matid[self.sim.model.body_geomadr[self.sim.model.body_name2id('object0')]] = 4
                    self.broken_object = 1.0
                # elif object_force > 0.0:
                #     self.sim.model.geom_matid[self.sim.model.body_geomadr[self.sim.model.body_name2id('object0')]] = 3
                #     self.broken_object = 0.0
                
                if self.n_actions == 4:
                    conc_stiffness_data = np.concatenate([[l_finger_force,
                                         r_finger_force]])/1000.0
                elif self.n_actions == 5:
                    conc_stiffness_data = np.concatenate([[l_finger_force,
                                         r_finger_force,
                                         self.prev_stiffness]])/1000.0
                elif self.n_actions == 6:
                    conc_stiffness_data = np.concatenate([[l_finger_force,
                                         r_finger_force,
                                         self.prev_stiffness,
                                         self.par_prev_stiffness + self.psv_prev_stiffness]])/1000.0
                elif self.n_actions == 7:
                    conc_stiffness_data = np.concatenate([[l_finger_force,
                                         r_finger_force,
                                         self.prev_stiffness,
                                         self.psv_prev_stiffness,
                                         self.par_prev_stiffness]])/1000.0
                
                obs = np.concatenate([
                    grip_pos, object_rel_pos.ravel(), gripper_state, goal_rel_pos.ravel(), conc_stiffness_data
                ])
                achieved_goal = np.concatenate([achieved_goal, conc_stiffness_data[:2]])
                
                self.prev_lforce = l_finger_force
                self.prev_rforce = r_finger_force
                self.prev_oforce = object_force
            else:
                finger_force = (self.sim.data.ctrl[2] - self.sim.data.sensordata[self.sim.model.sensor_name2id('robot0:Sjp_WRJ0')]) * self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('robot0:A_WRJ0'), 0] - self.sim.data.sensordata[self.sim.model.sensor_name2id('robot0:Sjp_WRJ0')] * self.sim.model.jnt_stiffness[self.sim.model.joint_name2id('robot0:WRJ0')]
                finger_force = self.prev_force + (finger_force - self.prev_force) * dt / 0.05
                
                object_force = self.prev_oforce + (self.sim.data.sensordata[self.sim.model.sensor_name2id('object_frc')] - self.prev_oforce) * dt / 0.1
                
                if self.n_actions == 3:
                    conc_stiffness_data = np.array([finger_force])/100.0
                elif self.n_actions == 4:
                    conc_stiffness_data = np.concatenate([[finger_force,
                                         self.prev_stiffness]])/100.0
                elif self.n_actions == 5:
                    conc_stiffness_data = np.concatenate([[finger_force,
                                         self.prev_stiffness,
                                         self.psv_prev_stiffness]])/100.0
                
                if (object_force > self.object_fragility):
                    self.sim.model.geom_matid[self.sim.model.body_geomadr[self.sim.model.body_name2id('object0')]:self.sim.model.body_geomadr[self.sim.model.body_name2id('object0')]+4] = 6
                    self.broken_object = 1.0
                # elif object_force > 0.0:
                #     self.sim.model.geom_matid[self.sim.model.body_geomadr[self.sim.model.body_name2id('object0')]:self.sim.model.body_geomadr[self.sim.model.body_name2id('object0')]+4] = 5
                #     self.broken_object = 0.0
                    
                obs = np.concatenate([
                    grip_pos, object_rel_pos.ravel(), [robot_qpos[2]], goal_rel_pos.ravel(), grip_velp, conc_stiffness_data
                ])
                achieved_goal = np.concatenate([achieved_goal, grip_velp*5.0, conc_stiffness_data[:1]])
                
                self.prev_force = finger_force
                self.prev_oforce = object_force
        else:
#            obs = np.concatenate([
#                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
#                object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel, goal_rel_pos.ravel()
#            ])
    
            obs = np.concatenate([
                grip_pos, object_rel_pos.ravel(), gripper_state, goal_rel_pos.ravel()
            ])
        # print(len(obs))
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link') if self.model_path.find('wall') == -1 else self.sim.model.body_name2id('basket')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        if self.model_path.find('wall') == -1:
            sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
            site_id = self.sim.model.site_name2id('target0')
            self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[site_id]
        else:
            self.sim.model.body_pos[self.sim.model.body_name2id('target')] = self.goal[:3]
#        print("SITE POS: {}, SITE OFFSET: {}, GOAL: {}".format(self.sim.model.site_pos[site_id], sites_offset[site_id], self.goal[:3]))
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        
        # reset the broken objects
        self.broken_object = False
        self.sim.model.geom_matid[self.sim.model.body_geomadr[self.sim.model.body_name2id('object0')]:self.sim.model.body_geomadr[self.sim.model.body_name2id('object0')]+4] = 5
        
        # reset stiffness
        self.prev_stiffness = self.max_stiffness
        self.psv_prev_stiffness = self.max_stiffness
        self.par_prev_stiffness = 0.0
        
        # reset forces
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_force = 0.0
        self.prev_oforce = 0.0
        
        self.remaining_timestep = 50

        # Randomize start position of object.
        if self.has_object and self.model_path.find('wall') == -1:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)#*np.array([1,0.3])
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        elif self.model_path.find('wall') != -1:
            initial_qpos = self.sim.data.get_joint_qpos('object0:joint').copy()
            initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
            initial_pos += np.array([np.random.random()*0.5-0.25, 0, 0])
            initial_qpos[:3] = initial_pos
            self.sim.data.set_joint_qpos('object0:joint', initial_qpos)
        if self.fragile_on:
            self.sim.model.body_mass[self.sim.model.body_name2id('object0')] = 2.0 if self.model_path.find('wall') == -1 else 0.1
            self.min_grip = self.sim.model.body_mass[self.sim.model.body_name2id('object0')] * self.grav_const / (2 * self.fric_mu)
            self.object_fragility = 6.0 * self.min_grip if self.model_path.find('wall') == -1 else 200.0
        elif self.has_object:
            initial_qpos = self.sim.data.get_joint_qpos('object0:joint').copy()
            initial_qpos[:3] += np.array([np.random.random()*0.5-0.25, 0, 0])
            self.sim.data.set_joint_qpos('object0:joint', initial_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object and self.model_path.find('wall') == -1:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air:# and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        elif self.has_object:
            offset = np.array([0.45*np.random.random()-0.24, 0.30*np.random.random(), 0.0])
            offset[2] = offset[1]
            goal = np.array([1.015, 0.78, -0.09]) + offset
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
            
        if self.model_path.find('wall') == -1: return np.concatenate([goal.copy(), [0.0, 0.0]])
        else: return np.concatenate([goal.copy(), [0.0, 0.0, 0.0, 0.0]])

    def _is_success(self, achieved_goal, desired_goal):
        if self.fragile_on:
            try: 
                d = goal_distance(achieved_goal[:,:3], desired_goal[:,:3])
                fragile_goal = np.linalg.norm(achieved_goal[:,3:] - desired_goal[:,3:], axis=-1)
            except: 
                d = goal_distance(achieved_goal[:3], desired_goal[:3])
                fragile_goal = np.linalg.norm(achieved_goal[3:] - desired_goal[3:])
        else: d = goal_distance(achieved_goal, desired_goal)
#        return (((d < self.distance_threshold).astype(np.float32) + np.float32(fragile_goal.sum(axis=-1)*1000.0 < self.object_fragility)) == 2.0).astype(np.float32) if self.fragile_on else (d < self.distance_threshold).astype(np.float32)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        if self.model_path.find('wall') == -1:
            self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
            self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
