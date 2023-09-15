import numpy as np
import math
import rospy 
# Gym 
from gymnasium import spaces
# from gym.envs.registration import register
# ROS msgs
from geometry_msgs.msg import Twist, Pose
# Gazebo interfaces
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

# ROS packages required
import tf.transformations

from auv_rl_gym.robot_envs.desistek_saga_env import DesistekSagaEnv

MIN_LINEAR_X = -10.0
MAX_LINEAR_X = 10.0

MIN_LINEAR_Z = -10.0
MAX_LINEAR_Z = 10.0

MIN_ANGULAR_Z = -10.0
MAX_ANGULAR_Z = 10.0

class AutoDocking(DesistekSagaEnv):
    def __init__(self):
        super(AutoDocking, self).__init__()
        self.init_intrinsic_parameters()
        self.init_hyperparameters()
        self.init_ROS_channels()
        
        self.init_member_variables()

        rospy.loginfo("Desistek_saga::AutoDocking: task started")

    def init_intrinsic_parameters(self):
        self.reward_range = (-np.inf, np.inf)
        
        self.action_space = self.set_action_space() 
        
        self.observation_space = self.set_obs_space_boundaries()
        
        self.star_time = None
        
    def set_action_space_old(self):
        """ 
        action[0]: x linear velocity
        action[1]: y linear velocity
        action[2]: z angular velocity (heading) 
        """
        return spaces.Box(low=np.array([MIN_LINEAR_X, MIN_LINEAR_Z, MIN_ANGULAR_Z]),
                                       high=np.array([MAX_LINEAR_X, MAX_LINEAR_Z, MAX_ANGULAR_Z]),
                                       dtype=np.float32)
        

    def set_action_space(self):
        """
        Action space to be symmetric and normalized between -1 and 1, as required by gymnasium.
        
        action[0]: x linear velocity
        action[1]: y linear velocity
        action[2]: z angular velocity (heading) 
        """
        return spaces.Box(low=np.array([-1.0, -1.0, -1.0]),
                        high=np.array([1.0, 1.0, 1.0]),
                        dtype=np.float32)

    # todo: change it to get from pose estimation during the experiments
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
        min_trans_error = -100000
        min_velocity_error = min_trans_error
        low = np.array([min_trans_error, min_trans_error, min_trans_error, -np.pi, min_velocity_error, 
                        min_velocity_error, min_velocity_error, min_velocity_error])
        high = low*-1
        
        return spaces.Box(low, high, dtype=np.float32)
    
    def init_hyperparameters(self):
        self.max_offset = rospy.get_param('/desistek_saga/episode/max_offset')
        self.min_offset = rospy.get_param('/desistek_saga/episode/min_offset')

        self.set_timeout(rospy.get_param('/desistek_saga/episode/timeout'))
        
        self.init_position_range_x = rospy.get_param('/desistek_saga/pose/init_position_range_x', [-5.0, 5.0])
        self.init_position_range_y = rospy.get_param('/desistek_saga/pose/init_position_range_y', [15.0, 25.0])
        self.init_position_range_z = rospy.get_param('/desistek_saga/pose/init_position_range_z', [-98.0, -95.0])
        self.init_orientation_range_yaw = rospy.get_param('/desistek_saga/pose/initial_orientation_range_yaw', [-0.5, 0.5])
        
        self.target_pose = self.set_target_pose()
        self.set_target_yaw = self.set_target_yaw()
        
    def init_member_variables(self):
        self.debug = {}
        self.debug['traveled_path'] = []
        self.debug['distances'] = []
        self.debug['heading errors'] = []
        self.random_respawn = True
        
    def set_timeout(self, timeout):
        self.timeout = timeout

    def set_target_pose(self):
        # Professor: Provide the desired goal pose. yes = using a local frame: either robot or camera
        position = rospy.get_param("/desistek_saga/pose/target_pose/position")
        orientation = rospy.get_param("/desistek_saga/pose/target_pose/orientation")

        target = Pose()
        target.position.x = position[0]
        target.position.y = position[1]
        target.position.z = position[2]
        
        quaternion = tf.transformations.quaternion_from_euler(
                        orientation[0],
                        orientation[1],
                        orientation[2],
                        )
        
        target.orientation.x = quaternion[0]
        target.orientation.y = quaternion[1]
        target.orientation.z = quaternion[2]
        target.orientation.w = quaternion[3]

        return target
    
    def set_target_yaw(self):
        return tf.transformations.euler_from_quaternion([
                self.target_pose.orientation.x,
                self.target_pose.orientation.y,
                self.target_pose.orientation.z,
                self.target_pose.orientation.w
                ])[2]
        
    def init_ROS_channels(self):
        self.respawn_service = '/gazebo/set_model_state'    

    def is_timeout(self):   
        elapsed_time = rospy.Time.now() - self.start_time
        
        if elapsed_time.to_sec() > self.timeout:
            rospy.loginfo("Desistek_saga::AutoDocking: Episode is over, timeout!")
            return True

        return False

    def distance_to_target(self, error):
        return np.linalg.norm(error)

    def is_inside_workspace(self, obs):
        position_error = np.array(obs[:3])
        distance_to_target = self.distance_to_target(position_error)
        
        if distance_to_target >= self.max_offset:
            rospy.loginfo("Desistek_saga::AutoDocking: Episode is over, agent out of workspace!")
            return False
        
        return True

    
    def create_pose_msg(self, translation, yaw):
        pose_msg = Pose()
        pose_msg.position.x = translation[0]
        pose_msg.position.y = translation[1]
        pose_msg.position.z = translation[2]
        
        quaternion = self.yaw_to_quaternion(yaw)

        pose_msg.orientation.x = quaternion[0]
        pose_msg.orientation.y = quaternion[1]
        pose_msg.orientation.z = quaternion[2]
        pose_msg.orientation.w = quaternion[3]

        return pose_msg
    
    def yaw_to_quaternion(self, yaw):
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        return quaternion
    
    def set_initial_poses(self, poses):
        self.random_respawn = False
        self.intial_positions = poses
        
    def calculate_heading_error(self, yaw, position_error):
        
        dx = position_error[0]
        dy = position_error[1]

        # Calculate the angle to the target in the global frame
        target_angle_global = math.atan2(dy, dx)

        # Calculate the relative angle from the robot's current yaw angle to the target
        relative_yaw_to_target = target_angle_global - yaw

        # Normalize the angle to be between -pi and pi
        heading_error = math.atan2(math.sin(relative_yaw_to_target), math.cos(relative_yaw_to_target))

        return heading_error

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
    def _set_action_old(self, action):
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.linear.z = action[1]
        cmd.angular.z = action[2]
        
        self._cmd_drive_pub.publish(cmd)
        self.last_action = action
        
        self.wait_time_for_execute_movement(1)
        
        #todo: add 1s and check if it works. Otherwise, implement turtlebot2 solution

    def _set_action(self, action):
        cmd = Twist()

        # Rescale the actions from normalized range to actual range
        cmd.linear.x = action[0] * (MAX_LINEAR_X - MIN_LINEAR_X) / 2.0 + (MAX_LINEAR_X + MIN_LINEAR_X) / 2.0
        cmd.linear.z = action[1] * (MAX_LINEAR_Z - MIN_LINEAR_Z) / 2.0 + (MAX_LINEAR_Z + MIN_LINEAR_Z) / 2.0
        cmd.angular.z = action[2] * (MAX_ANGULAR_Z - MIN_ANGULAR_Z) / 2.0 + (MAX_ANGULAR_Z + MIN_ANGULAR_Z) / 2.0

        self._cmd_drive_pub.publish(cmd)
        self.last_action = action

        self.wait_time_for_execute_movement(1)
        
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
        odom_pose = odom.pose.pose
        position_error = [self.target_pose.position.x - odom_pose.position.x,
                          self.target_pose.position.y - odom_pose.position.y,
                          self.target_pose.position.z - odom_pose.position.z]
                          
        current_yaw = tf.transformations.euler_from_quaternion([
            odom_pose.orientation.x,
            odom_pose.orientation.y,
            odom_pose.orientation.z,
            odom_pose.orientation.w
        ])[2]
        
        rospy.loginfo ("Desistek_saga::AutoDocking: Current Yaw = " + str(current_yaw)) 
                          
        heading_error = self.calculate_heading_error(current_yaw, position_error)
        
        linear_velocities = [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z]
        angular_velocity_z = odom.twist.twist.angular.z
        
        obs = position_error + [heading_error] + linear_velocities + [angular_velocity_z]
        
        return np.array(obs, dtype=np.float32)
    
    def _is_terminated(self, observations):
        
        position_error = np.array(observations[:3])
        distance_to_target = self.distance_to_target(position_error)
        
        if distance_to_target <= self.min_offset:
            rospy.loginfo("Desistek_saga::AutoDocking: Episode is over, agent reached the goal!")
            return True
        
        return False 
        
    def _is_truncated(self, observations):
        return (self.is_timeout() or (not self.is_inside_workspace(observations)))
  
    def _compute_reward(self, observations, terminated):
        heading_error = np.abs(observations[3])
        
        position_error = np.array(observations[:3])
        euclidean_distance = np.linalg.norm(position_error)
        
        rospy.loginfo ("########")
        rospy.loginfo ("Desistek_saga::AutoDocking: Episode = " + str(self.episode_num))
        rospy.loginfo ("Desistek_saga::AutoDocking: Elapsed time = " + str((rospy.Time.now() - self.start_time).to_sec()))
        rospy.loginfo ("Desistek_saga::AutoDocking: Position Error = " + str(position_error)) 
        rospy.loginfo ("Desistek_saga::AutoDocking: Heading Error = " + str(heading_error)) 
        rospy.loginfo ("Desistek_saga::AutoDocking: Distance to target = " + str(euclidean_distance)) 
        reward = -(heading_error + euclidean_distance)
        rospy.loginfo ("Desistek_saga::AutoDocking: Step reward = " + str(reward))

        return float(reward)

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
            
            if not self.random_respawn and self.intial_positions:
                pose = self.intial_positions.pop(0)
                x = pose[0]
                y = pose[1]
                z = pose[2]
                yaw = pose[3]
                rospy.loginfo("Using provided initial poses for respawning")
            else:
                x = np.random.uniform(self.init_position_range_x[0], self.init_position_range_x[1])
                y = np.random.uniform(self.init_position_range_y[0], self.init_position_range_y[1])
                z = np.random.uniform(self.init_position_range_z[0], self.init_position_range_z[1])
                yaw = np.random.uniform(self.init_orientation_range_yaw[0], self.init_orientation_range_yaw[1])

                rospy.loginfo("Using random initial poses for respawning")

            pose_msg = self.create_pose_msg([x, y, z], yaw)

            state_msg.pose = pose_msg
            set_model_state(state_msg)
            
            rospy.loginfo("Respawning desistek_saga at position: x: %s, y: %s, z: %s with yaw: %s" % 
                            (x, y, z, yaw))

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
        
if __name__ == '__main__':
    rospy.init_node('desistek_saga_auto_docking', anonymous=True, log_level=rospy.DEBUG)
    env = AutoDocking()
    rospy.spin()