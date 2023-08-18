import numpy as np

import rospy 
# Gym 
from gym import spaces
from gym.envs.registration import register
# ROS msgs
from geometry_msgs.msg import Twist, Pose
# Gazebo interfaces
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

# ROS packages required
import tf.transformations

from auv_rl_gym.robot_envs.desistek_saga_env import DesistekSagaEnv

TIMESTEP_LIMIT_PER_EPISODE = 10000 

MIN_LINEAR_X = -10.0
MAX_LINEAR_X = 10.0

MIN_LINEAR_Z = -10.0
MAX_LINEAR_Z = 10.0

MIN_ANGULAR_Z = -10.0
MAX_ANGULAR_Z = 10.0

register(
        id='DesistekSagaAutoDocking-v0',
        entry_point='auv_rl_gym.task_envs.desistek_saga.auto_docking:AutoDocking',
        timestep_limit=TIMESTEP_LIMIT_PER_EPISODE,
    )

class AutoDocking(DesistekSagaEnv):
    def ___init__(self):
        self.init_intrinsic_parameters()
        self.init_hyperparameters()
        self.init_ROS_channels()
        
        super(AutoDocking, self).__init__()

        rospy.loginfo("Desistek_saga::AutoDocking: task started")

    def init_intrinsic_parameters(self):
        self.reward_range = (-np.inf, np.inf)
        
        self.action_space = self.set_action_space() 
        
        self.observation_space = self.set_obs_space_boundaries()
        
        self.star_time = None
        
        rospy.loginfo("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.loginfo("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

    def set_action_space(self):
        """ 
        action[0]: x linear velocity
        action[1]: y linear velocity
        action[2]: z angular velocity (heading) 
        """
        return spaces.Box(low=np.array( [MIN_LINEAR_X, MIN_LINEAR_Z, MIN_ANGULAR_Z]),
                                       high=np.array([MAX_LINEAR_X, MAX_LINEAR_Z, MAX_ANGULAR_Z]),
                                       dtype=np.float32)

    def set_obs_space_boundaries(self):
        """ 
        obs[0]: Translation error in the x-direction
        obs[1]: Translation error in the y-direction
        obs[2]: Translation error in the z-direction
        obs[3]: Heading error
        obs[4]: x linear velocity from the last step
        obs[5]: y linear velocity from the last step
        obs[6]: z linear velocity from the last step
        obs[7]: z angular velocity from the last step 
        """
        low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        high = low*-1
        
        return spaces.Box(low, high, dtype=np.float32)
    
    def init_hyperparameters(self):
        self.max_offset = rospy.get_param('/desistek_saga/episode/max_offset')
        self.min_offset = rospy.get_param('/desistek_saga/episode/min_offset')

        self.timeout = rospy.get_param('/desistek_saga/episode/timeout')
        
        self.target_pose = self.set_target_pose()
        self.set_tartget_yaw = self.set_target_yaw()
        self.initial_position, self.initial_orientation = self.set_initial_pose()
        
    def set_target_pose(self):
        # Professor: Provide the desired goal pose. yes = using a local frame: either robot or camera
        position = rospy.get_param("/desistek_saga/pose/target_pose/position")
        orientation = rospy.get_param("/desistek_saga/pose/target_pose/orientation")

        target = Pose()
        target.pose.pose.position.x = position[0]
        target.pose.pose.position.y = position[1]
        target.pose.pose.position.z = position[2]
        
        target.pose.pose.orientation.x = orientation[0]
        target.pose.pose.orientation.y = orientation[1]
        target.pose.pose.orientation.z = orientation[2]
        target.pose.pose.orientation.w = orientation[3]

        return target
    
    def set_tartget_yaw(self):
        return tf.transformations.euler_from_quaternion([
                self.target_pose.pose.pose.orientation.x,
                self.target_pose.pose.pose.orientation.y,
                self.target_pose.pose.pose.orientation.z,
                self.target_pose.pose.pose.orientation.w
                ])[2]
        
    def set_initial_pose(self):
        position = rospy.get_param('/desistek_saga/pose/initial_position')
        orientation = rospy.get_param('/desistek_saga/pose/initial_orientation')
        
        self.initial_pose_msg = self.create_pose_msg(position, orientation)
        
        return position, orientation
        
    def init_ROS_channels(self):
        self.respawn_service = '/gazebo/set_model_state'    

    def is_timeout(self):   
        elapsed_time = rospy.Time.now() - self.start_time
        return elapsed_time > self.timeout

    def is_inside_workspace(self, obs):
        position_error = np.array(obs[:3])
        distance_to_target = np.linalg.norm(position_error)
        
        is_near_target = distance_to_target < self.min_offset
        
        is_within_boundary = distance_to_target < self.max_offset
        
        return is_near_target and is_within_boundary
    
    def create_pose_msg(self, translation, euler):
        pose_msg = Pose()
        pose_msg.position.x = translation[0]
        pose_msg.position.y = translation[1]
        pose_msg.position.z = translation[2]
        
        quaternion = tf.transformations.quaternion_from_euler(euler)

        pose_msg.orientation.x = quaternion[0]
        pose_msg.orientation.y = quaternion[1]
        pose_msg.orientation.z = quaternion[2]
        pose_msg.orientation.w = quaternion[3]

        return pose_msg
    
    def yaw_to_quaternion(self, yaw):
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        return quaternion

    # Methods that the this class implements
    # They will be used in RobotGazeboEnv GrandParentClass and defined in the DesistekSagaEnv
    # ----------------------------
    # The following methods are required to compose the step() method, 
    # required for the stable baselines interface
    """ 
    def step(self, action):
        self._set_action(action) (overriden)
        obs = self._get_obs() (overriden)
        done = self._is_done(obs) (overriden)
        info = {}
        reward = self._compute_reward(obs, done) (overriden)
    """
    # ----------------------------
    def _set_action(self, action):
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.linear.y = action[1]
        cmd.angular.z = action[2]
        
        self._cmd_drive_pub.publish(cmd)
        self.last_action = action

    def _get_obs(self):
        """ 
        obs[0]: Translation error in the x-direction
        obs[1]: Translation error in the y-direction
        obs[2]: Translation error in the z-direction
        obs[3]: Heading error
        obs[4]: x linear velocity from the last step
        obs[5]: y linear velocity from the last step
        obs[6]: z linear velocity from the last step
        obs[7]: z angular velocity from the last step 
        """
        odom = self._get_odom()
        position_error = [self.target_pose.pose.pose.position[0] - odom.pose.pose.position.x,
                          self.target_pose.pose.pose.position[1] - odom.pose.pose.position.y,
                          self.target_pose.pose.pose.position[2] - odom.pose.pose.position.z]
                          
        current_yaw = tf.transformations.euler_from_quaternion([
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w
        ])[2]
                          
        heading_error = self.set_tartget_yaw - current_yaw 
        
        linear_velocities = [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z]
        angular_velocity_z = odom.twist.twist.angular.z
        
        return position_error + [heading_error] + linear_velocities + [angular_velocity_z]
    
    def _is_done(self, observations):
        return self.is_timeout() or self.is_inside_workspace(observations)

  
    def _compute_reward(self, observations, done):
        heading = np.abs(observations[3])
        
        position_error = np.array(observations[:3])
        euclidean_distance = np.linalg.norm(position_error)
        
        # Compute the reward
        reward = -(heading + euclidean_distance)
        
        return reward

    # Following the methods required for reseting the environment are implemented
    # The interface method called in reset(), which returns the current observation
    # ----------------------------
    """
    def reset(self):
        self._reset_sim()
        self._init_env_variables() (overriden)
        self._update_episode()
        obs = self._get_obs() (overriden)
        return obs

    def _reset_sim(self):
        self._check_all_systems_ready()
        self._set_init_pose() (overriden)
    """
    def _set_init_pose(self):
        rospy.wait_for_service(self.respawn_service)
        try:
            set_model_state = rospy.ServiceProxy(self.respawn_service, SetModelState)
            state_msg = ModelState()
            state_msg.model_name = "desistek_saga"

            pose_msg = self.initial_pose_msg

            state_msg.pose = pose_msg
            set_model_state(state_msg)

        except rospy.ServiceException as e:
            rospy.loginfo("Failed to desistek_saga state: %s" % e)

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        self.cumulated_reward = 0.0
        self._episode_done = False
        self.start_time = rospy.Time.now()