# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:23:34 2022

@author: keerthi.sagar

Launch a Serial/Tree robot with URDF

1. Instantiate Robot parameter Server here
2. Publish and Subscribe to the robot joints
3. Call IK solver based on Scipy.optimize - Constraint Function/ Objective Function (Maximize Capacity Margin)



File is inspired by the Architecture of Robot Solver of Antun Skuric

https://github.com/askuric/polytope_vertex_search/blob/master/ROS_nodes/panda_capacity/scripts/capacity/robot_solver.py





"""


## Library import
############# ROS Dependencies #####################################
import rospy
from geometry_msgs import msg
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped,WrenchStamped, PointStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy, JointState, PointCloud 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf_conversions as tf_c
from tf2_ros import TransformBroadcaster

#### Polytope Plot - Dependency #####################################

## Polygon plot for ROS - Geometry message
from jsk_recognition_msgs.msg import PolygonArray, SegmentArray
from geometry_msgs.msg import Polygon, PolygonStamped, Point32


from rospygradientpolytope.visual_polytope import velocity_polytope,desired_polytope,velocity_polytope_with_estimation
from rospygradientpolytope.polytope_ros_message import create_polytopes_msg, create_polygon_msg,create_capacity_vertex_msg, create_segment_msg


#################### Linear Algebra ####################################################

from numpy.core.numeric import cross
from numpy import matrix,matmul,transpose,isclose,array,rad2deg,abs,vstack,hstack,shape,eye,zeros

from numpy.linalg import norm,det
from math import atan2, pi,asin,acos


from re import T

################# URDF Parameter Server ##################################################

from urdf_parser_py.urdf import URDF

from pykdl_utils.kdl_kinematics import KDLKinematics



## Mutex operator
from threading import Lock

# getting the node namespace
namespace = rospy.get_namespace()


from tf.transformations import quaternion_matrix


mutex = Lock()

# Import URDF of the robot - # todo the param file for fetching the URDF without file location
## Launching the Robot here - Roslaunching an external launch file of the robot


# loading the root urdf from robot_description parameter
# Get the URDF from the parameter server
# Launch the robot 
robot_description = URDF.from_parameter_server() 

### For Universal Robot - UR5
base_link = "base_link"

tip_link = "wrist_3_link"


kdl_kin = KDLKinematics(robot_description , base_link, tip_link)

# Build the tree here from the URDF parser file from the file location
'''
PyKDL based chain - Only for faster/real-time implementation
build_ok, kdl_tree = urdf.treeFromFile('/home/imr/catkin_telebot_ws/src/kuka_experimental/kuka_kr4_support/urdf/kr4r600.urdf')

if  build_ok == True:
	print('KDL chain built successfully !!')
else:
	print('KDL chain unsuccessful')
'''


##############################

## PyKDL_Util here
pykdl_util_kin = KDLKinematics(robot_urdf , "base_link", "tool0")

# URDF parsing an kinematics - Using pykdl_utils : For faster version maybe use PyKDL directly without wrapper
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics

# Build kinematic chain here

class LaunchRobot():

    def __init__(self):
		# publish plytope --- Actual Polytope - Publisher
        self.publish_velocity_polytope = rospy.Publisher("/available_velocity_polytope", PolygonArray, queue_size=100)
        self.publish_desired_polytope = rospy.Publisher("/desired_velocity_polytope", PolygonArray, queue_size=100)
        self.publish_capacity_margin_polytope = rospy.Publisher("/capacity_margin_polytope", PolygonArray, queue_size=100)		
        self.publish_vertex_capacity = rospy.Publisher("/capacity_margin_vertex", PointStamped, queue_size=1)
        self.publish_vertex_proj_capacity = rospy.Publisher("/capacity_margin_proj_vertex", PointStamped, queue_size=1)
        self.publish_capacity_margin_actual = rospy.Publisher("/capacity_margin_actual", SegmentArray, queue_size=1)
		

		# publish plytope --- Estimated Polytope - Publisher
		
        self.publish_velocity_polytope_est = rospy.Publisher("/available_velocity_polytope_est", PolygonArray, queue_size=100)		
        self.publish_capacity_margin_polytope_est = rospy.Publisher("/capacity_margin_polytope_est", PolygonArray, queue_size=100)		
        self.publish_vertex_proj_capacity_est = rospy.Publisher("/capacity_margin_proj_vertex_est", PointStamped, queue_size=1)
        self.publish_capacity_margin_actual_est = rospy.Publisher("/capacity_margin_actual_est", SegmentArray, queue_size=1)

        self.cartesian_desired_vertices = 3*array([[0.20000, 0.50000, 0.50000],
								[0.50000, -0.10000, 0.50000],
								[0.50000, 0.50000, -0.60000],
								[0.50000, -0.10000, -0.60000],
								[-0.30000, 0.50000, 0.50000],
								[-0.30000, -0.10000, 0.50000],
								[-0.30000, 0.50000, -0.60000],
								[-0.30000, -0.10000, -0.60000]])
        
        self.pub_rate = 250 #Hz

        self.q_upper_limit = [robot_description.joint_map[i].limit.upper - 0.07 for i in self.robot_joint_names]
        self.q_lower_limit = [robot_description.joint_map[i].limit.lower + 0.07 for i in self.robot_joint_names]

        self.qdot_limit = [robot_description.joint_map[i].limit.velocity for i in self.robot_joint_names]

		








if __name__ == '__main__':
	print("geomagic to KUKA controller start up v2 File\n")
	controller = LaunchRobot()
	#controller.start()
	rospy.spin()