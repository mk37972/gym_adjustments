import numpy as np

from gym.envs.robotics import rotations, robot_env, utils

R_j = np.matrix([[0.01575,0],
                  [-0.01575, 0.01575]])
R_j_inv = np.linalg.inv(R_j)
R_j_L = np.matrix([[0.01575,0],
                  [0.01575, 0.01575]])
R_j_inv_L = np.linalg.inv(R_j_L)
R_e = np.matrix([[0.0034597,0],
                  [0, 0.0034597]])
L1 = 0.1
L2 = 0.075

Ksc = 700

Rm = 0.0285

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class NuFingersEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, target_range,
        distance_threshold, initial_qpos, reward_type, n_actions, pert_type
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
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.broken_table = False
        self.broken_object = False
        self.max_stiffness = 1.0
        self.prev_stiffness = self.max_stiffness
        self.prev_stiffness_limit = self.max_stiffness
        self.object_fragility = 0.0
        self.min_grip = 0.0
        self.fric_mu = 0.2
        self.grav_const = 9.81
        self.prev_force = 0.0
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_oforce = 0.0
        self.previous_input = 0
        self.remaining_timestep = 200
        self.des_Fp_R = np.array([[0.0],[0.0]])
        self.des_Fp_L = np.array([[0.0],[0.0]])
        self.goal_dim = 1

        super(NuFingersEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, initial_qpos=initial_qpos, n_actions=n_actions)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        try: 
            d = goal_distance(achieved_goal[:,:self.goal_dim], goal[:,:self.goal_dim])
            fragile_goal = np.linalg.norm((achieved_goal[:,self.goal_dim:] - goal[:,self.goal_dim:])*((achieved_goal[:,self.goal_dim:] - goal[:,self.goal_dim:]) < 0), axis=-1)
        except: 
            d = goal_distance(achieved_goal[:self.goal_dim], goal[:self.goal_dim])
            fragile_goal = np.linalg.norm((achieved_goal[self.goal_dim:] - goal[self.goal_dim:])*((achieved_goal[self.goal_dim:] - goal[self.goal_dim:]) < 0), axis=-1)
        
        return -(d > self.distance_threshold).astype(np.float32) - np.float32(fragile_goal) * 2.0
        

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.sim.forward()

    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope
        # relative L action, TH action (right), relative L action, TH action (left)
        # np.array([des_l_R - p_R[0,0], (des_th_R - p_R[1,0]) * np.pi / 180.0, des_l_L - p_L[0,0], (des_th_L - p_L[1,0]) * np.pi / 180.0, 0.0, 0.0])
        pos_ctrl_R, pos_ctrl_L = action[:2], action[2:4]
        # pos_ctrl_R *= np.array([0.01, np.pi/10.0])
        # pos_ctrl_L *= np.array([0.01, np.pi/10.0])
        stiffness_ctrl = 0.0
        stiffness_limit = 0.0

        # pos_ctrl_R = np.clip(pos_ctrl_R, np.array([-0.2, -0.1]), np.array([0.2, 0.1]))
        # pos_ctrl_L = np.clip(pos_ctrl_L, np.array([-0.2, -0.1]), np.array([0.2, 0.1]))
        
        if action.shape[0] > 4:
            stiffness_limit = 0.2 * self.max_stiffness * action[5]
            
            self.prev_stiffness_limit += stiffness_limit
            self.prev_stiffness_limit = np.max([np.min([self.prev_stiffness_limit, self.max_stiffness]), self.max_stiffness / 25.0])
            
            stiffness_ctrl = 0.2 * self.max_stiffness * action[4]
            
            self.prev_stiffness += stiffness_ctrl
            self.prev_stiffness = np.max([np.min([self.prev_stiffness, self.prev_stiffness_limit]), 0.0])
        
        Pc_R = np.array([-0.0635, 0.127])
        Pc_L = np.array([0.0635, 0.127])
        [xR, yR, zR] = self.sim.data.site_xpos[self.sim.model.site_name2id('Right_fingertip')]
        [xL, yL, zL] = self.sim.data.site_xpos[self.sim.model.site_name2id('Left_fingertip')]
        
        P_R = np.array([yR - 0.0889, 0.0873 - xR])
        P_L = np.array([yL + 0.0889, 0.0873 - xL])
        [xR, yR] = P_R
        [xL, yL] = P_L
        
        Prel_R = Pc_R - P_R
        Prel_L = Pc_L - P_L
        l_R = np.sqrt(Prel_R[0]*Prel_R[0] + Prel_R[1]*Prel_R[1])
        l_L = np.sqrt(Prel_L[0]*Prel_L[0] + Prel_L[1]*Prel_L[1])
        p_R = np.array([[l_R],[np.arctan2(-Prel_R[1],-Prel_R[0])]])
        p_L = np.array([[l_L],[np.arctan2(Prel_L[1],Prel_L[0])]])
        
        r = np.array([[self.prev_stiffness], [1.0]])
        des_l_R = p_R[0,0] + pos_ctrl_R[0]
        des_th_R = p_R[1,0] + pos_ctrl_R[1]
        # print(pos_ctrl_R)
        des_p_R = np.array([[np.min([np.max([des_l_R, -0.06]), 0.06])], [np.min([np.max([des_th_R, -np.pi/2.0]), np.pi/2.0])]])#0.7854
        
        des_l_L = p_L[0,0] + pos_ctrl_L[0]
        des_th_L = p_L[1,0] + pos_ctrl_L[1]
        des_p_L = np.array([[np.min([np.max([des_l_L, -0.06]), 0.06])], [np.min([np.max([des_th_L, -np.pi/2.0]), np.pi/2.0])]])#0.7854
        # print(p_L)
        Rj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_R')]] + self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('FakeJoint_1_R')]]],
                       [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_R')]] + self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('FakeJoint_2_R')]]]])
        # print(Rj)
        Lj = np.array([[self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_1_L')]] + self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('FakeJoint_1_L')]]],
                       [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Joint_2_L')]] + self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('FakeJoint_2_L')]]]])
        Jp_R = np.matrix([[-Prel_R[0]/l_R, -Prel_R[1]/l_R],[Prel_R[1]/l_R/l_R, -Prel_R[0]/l_R/l_R]])
        Jp_L = np.matrix([[-Prel_L[0]/l_L, -Prel_L[1]/l_L],[Prel_L[1]/l_L/l_L, -Prel_L[0]/l_L/l_L]])
        Jp_inv_R = np.matrix([[Jp_R[1,1] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0]), -Jp_R[0,1] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0])], [-Jp_R[1,0] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0]), Jp_R[0,0] / (Jp_R[0,0]*Jp_R[1,1] - Jp_R[0,1]*Jp_R[1,0])]])
        Jp_inv_L = np.matrix([[Jp_L[1,1] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0]), -Jp_L[0,1] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0])], [-Jp_L[1,0] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0]), Jp_L[0,0] / (Jp_L[0,0]*Jp_L[1,1] - Jp_L[0,1]*Jp_L[1,0])]])
        J_R = np.matrix([[-yR, L2 * np.cos(Rj[0,0]-Rj[1,0])], 
                         [xR, L2 * np.sin(Rj[0,0]-Rj[1,0])]])
        J_L = np.matrix([[-yL, -L2 * np.cos(Lj[0,0]+Lj[1,0])], 
                         [xL, -L2 * np.sin(Lj[0,0]+Lj[1,0])]])
        J_inv_R = np.matrix([[J_R[1,1] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0]), -J_R[0,1] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0])], [-J_R[1,0] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0]), J_R[0,0] / (J_R[0,0]*J_R[1,1] - J_R[0,1]*J_R[1,0])]])
        J_inv_L = np.matrix([[J_L[1,1] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0]), -J_L[0,1] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0])], [-J_L[1,0] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0]), J_L[0,0] / (J_L[0,0]*J_L[1,1] - J_L[0,1]*J_L[1,0])]])
        max_kj_R = np.transpose(R_j) * np.matrix([[self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T1_R')], 0],[0, self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T2_R')]]]) * R_j
        max_kj_L = np.transpose(R_j_L) * np.matrix([[self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T1_L')], 0],[0, self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T2_L')]]]) * R_j_L
        max_k_R = np.transpose(J_inv_R) * max_kj_R * J_inv_R
        max_k_L = np.transpose(J_inv_L) * max_kj_L * J_inv_L
        max_kp_R = np.transpose(Jp_inv_R) * max_k_R * Jp_inv_R
        max_kp_L = np.transpose(Jp_inv_L) * max_k_L * Jp_inv_L
        self.des_Fp_R = max_kp_R * (r * (des_p_R - p_R))
        self.des_Fp_L = max_kp_L * (r * (des_p_L - p_L))
        des_F_R = np.transpose(Jp_R) * self.des_Fp_R
        des_F_L = np.transpose(Jp_L) * self.des_Fp_L
        des_tau_R = np.transpose(J_R) * des_F_R
        des_tau_L = np.transpose(J_L) * des_F_L
        des_mR = ((np.matrix([[1/self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T1_R')], 0],[0, 1/self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T2_R')]]]) * np.transpose(R_j_inv)*des_tau_R) + R_j * Rj) / Rm 
        des_mL = ((np.matrix([[1/self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T1_L')], 0],[0, 1/self.sim.model.tendon_stiffness[self.sim.model.tendon_name2id('T2_L')]]]) * np.transpose(R_j_inv_L)*des_tau_L) + R_j_L * Lj) / Rm
        
        self.sim.data.ctrl[0] = des_mL[0,0]
        self.sim.data.ctrl[1] = des_mL[1,0]
        self.sim.data.ctrl[2] = des_mR[0,0]
        self.sim.data.ctrl[3] = des_mR[1,0]

    def _get_obs(self):
        # positions
        self.remaining_timestep -= 1
        l_finger_force = self.prev_lforce + (self.des_Fp_R[0,0] - self.prev_lforce) * 0.004 / 0.05
        r_finger_force = self.prev_rforce + (self.des_Fp_L[0,0] - self.prev_rforce) * 0.004 / 0.05  
        
        Pc_R = np.array([-0.0635, 0.127])
        Pc_L = np.array([0.0635, 0.127])
        [xR, yR, zR] = self.sim.data.site_xpos[self.sim.model.site_name2id('Right_fingertip')]
        [xL, yL, zL] = self.sim.data.site_xpos[self.sim.model.site_name2id('Left_fingertip')]
        
        P_R = np.array([yR - 0.0889, 0.0873 - xR])
        P_L = np.array([yL + 0.0889, 0.0873 - xL])
        [xR, yR] = P_R
        [xL, yL] = P_L
        
        Prel_R = Pc_R - P_R
        Prel_L = Pc_L - P_L
        l_R = np.sqrt(Prel_R[0]*Prel_R[0] + Prel_R[1]*Prel_R[1])
        l_L = np.sqrt(Prel_L[0]*Prel_L[0] + Prel_L[1]*Prel_L[1])
        p_R = np.array([[l_R],[np.arctan2(-Prel_R[1],-Prel_R[0])]])
        p_L = np.array([[l_L],[np.arctan2(Prel_L[1],Prel_L[0])]])
        
        obj_rot = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('Sensor_joint')]]
        observation = np.array([p_R[0,0] * 10 - 1.0, p_L[0,0] * 10 - 1.0, p_R[1,0], p_L[1,0],
                                (obj_rot - p_R[1,0]), (obj_rot - p_L[1,0]), 
                                (self.goal[0] - obj_rot),
                                l_finger_force * 0.1, r_finger_force * 0.1, 
                                self.prev_stiffness, self.prev_stiffness_limit])
        
        modified_obs = dict(observation=observation, achieved_goal=np.array([obj_rot, l_finger_force * 0.1, r_finger_force * 0.1]), desired_goal = self.goal)
        
        self.prev_lforce = l_finger_force
        self.prev_rforce = r_finger_force
        
        return modified_obs

    def _viewer_setup(self):
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = 32.
        self.viewer.cam.elevation = -30.

    def _render_callback(self):
        # Visualize target.
        self.sim.model.site_quat[self.sim.model.site_name2id('target0')] = np.array([np.cos(self.goal[0]/2.0), np.sin(self.goal[0]/2.0), 0.0, 0.0])
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        
        # reset the broken objects
        self.broken_object = False
        # self.sim.model.geom_rgba[-1]
        
        # reset stiffness
        self.prev_stiffness = self.max_stiffness
        self.prev_stiffness_limit = self.max_stiffness
        
        # reset forces
        self.prev_lforce = 0.0
        self.prev_rforce = 0.0
        self.prev_oforce = 0.0
        
        self.remaining_timestep = 200

        # Randomize start position of object.

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.np_random.uniform(-self.target_range, self.target_range, size=1)
        return np.concatenate([goal.copy(), [0.0, 0.0]])

    def _is_success(self, achieved_goal, desired_goal):
        try: 
            d = goal_distance(achieved_goal[:,:self.goal_dim], desired_goal[:,:self.goal_dim])
        except: 
            d = goal_distance(achieved_goal[:self.goal_dim], desired_goal[:self.goal_dim])
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        for _ in range(10):
            self.sim.step()

    def render(self, mode='human', width=500, height=500):
        return super(NuFingersEnv, self).render(mode, width, height)
