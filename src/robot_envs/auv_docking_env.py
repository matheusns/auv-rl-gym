import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from nav_msgs.msg import Odometry

class AuvDockingEnv(robot_gazebo_env.RobotGazeboEnv):
    def __init__(self):
        rospy.logdebug("Start AuvDockingEnv INIT...")
        
        self.init_attributes()
        self.init_ROS_attributes()

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(AuvDockingEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")

        
        # rospy.logdebug("AuvDockingEnv unpause 1...")
        # self.gazebo.unpauseSim()
        
        # self._check_all_systems_ready()
        # self._check_all_publishers_ready()

        # self.gazebo.pauseSim()
        
        # rospy.logdebug("Finished AuvDockingEnv INIT...")

    def init_attributes(self):
        
        self.controllers_list = []
        self.robot_name_space = "" 
        self.publishers_array = []
        
    def init_ROS_attributes(self):
        if not rospy.has_param('/odom_topic') or not rospy.has_param('/velocity_topic'):
            rospy.loginfo("Required parameters not set! Please ensure both velocity_topic and odom_topic are set.")
            return

        # Read parameters from the ROS parameter server
        velocity_topic_name = rospy.get_param('/publisher_topic_name', '/default_publisher_topic')
        odom_topic_name = rospy.get_param('/subscriber_topic_name', '/default_subscriber_topic')

        # Print the parameters
        rospy.loginfo("Publisher Topic Name: %s", velocity_topic_name)
        rospy.loginfo("Subscriber Topic Name: %s", odom_topic_name)
            
        # self._cmd_drive_pub = rospy.Publisher('/cmd_drive', UsvDrive, queue_size=1)
        # self.publishers_array.append(self._cmd_drive_pub)
        # rospy.Subscriber("/wamv/odom", Odometry, self._odom_callback)
        

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("WamvEnv check_all_systems_ready...")
        self._check_all_sensors_ready()
        rospy.logdebug("END WamvEnv _check_all_systems_ready...")
        return True

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /wamv/odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/wamv/odom", Odometry, timeout=1.0)
                rospy.logdebug("Current /wamv/odom READY=>")

            except:
                rospy.logerr("Current /wamv/odom not ready yet, retrying for getting odom")
        return self.odom
    
    def _odom_callback(self, data):
        self.odom = data
    
    def _check_all_publishers_ready(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rospy.logdebug("START ALL SENSORS READY")
        for publisher_object in self.publishers_array:
            self._check_pub_connection(publisher_object)
        rospy.logdebug("ALL SENSORS READY")

    def _check_pub_connection(self, publisher_object):

        rate = rospy.Rate(10)  # 10hz
        while publisher_object.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to publisher_object yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("publisher_object Publisher Connected")

        rospy.logdebug("All Publishers READY")
        
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def set_propellers_speed(self, right_propeller_speed, left_propeller_speed, time_sleep=1.0):
        """
        It will set the speed of each of the two proppelers of wamv.
        """
        i = 0
        for publisher_object in self.publishers_array:
          usv_drive_obj = UsvDrive()
          usv_drive_obj.right = right_propeller_speed
          usv_drive_obj.left = left_propeller_speed
          
          rospy.logdebug("usv_drive_obj>>"+str(usv_drive_obj))
          publisher_object.publish(usv_drive_obj)
          i += 1
        self.wait_time_for_execute_movement(time_sleep)
    
    def wait_time_for_execute_movement(self, time_sleep):
        """
        Because this Wamv position is global, we really dont have
        a way to know if its moving in the direction desired, because it would need
        to evaluate the diference in position and speed on the local reference.
        """
        time.sleep(time_sleep)
    
    def get_odom(self):
        return self.odom

