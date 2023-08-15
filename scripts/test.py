#!/usr/bin/env python
import rospy

from auv_rl_gym.robot_envs.auv_docking_env import AuvDockingEnv

rospy.init_node('test_gym', anonymous=True, log_level=rospy.INFO)

test = AuvDockingEnv()