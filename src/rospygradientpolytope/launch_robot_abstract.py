#!/usr/bin/env python3
## 
###

#### Old Dependencies


#!/usr/bin/env python
# Library import
############# ROS Dependencies #####################################
import rospy
import os
from geometry_msgs import msg
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped, WrenchStamped, PointStamped
from std_msgs.msg import Bool, Float32,Int16,String

from sensor_msgs.msg import Joy, JointState, PointCloud
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial import distance as dist_scipy
from numpy import sum,eye

from numpy import size


import tf_conversions as tf_c
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers.interactive_marker_server import *
from geometry_msgs.msg import Pose, Point
#### Polytope Plot - Dependency #####################################

# Polygon plot for ROS - Geometry message
from jsk_recognition_msgs.msg import PolygonArray, SegmentArray
import matplotlib.pyplot as plt

from geometry_msgs.msg import Polygon, PolygonStamped, Point32, Pose
import time
import threading


from rospygradientpolytope.visual_polytope import velocity_polytope, desired_polytope, velocity_polytope_with_estimation
from rospygradientpolytope.polytope_ros_message import create_polytopes_msg, create_polygon_msg, create_capacity_vertex_msg, create_segment_msg
from rospygradientpolytope.polytope_functions import get_polytope_hyperplane, get_capacity_margin
#from rospygradientpolytope.polytope_gradient_functions_optimized import Gamma_hat_gradient
from rospygradientpolytope.polytope_gradient_functions import Gamma_hat_gradient,Gamma_hat_gradient_dq

from rospygradientpolytope.sawyer_functions import jacobianE0, position_70
from rospygradientpolytope.robot_functions import getHessian, getJ_pinv
from rospygradientpolytope.linearalgebra import check_ndarray

#################### Linear Algebra ####################################################

from numpy.core.numeric import cross
from numpy import matrix, matmul, transpose, isclose, array, rad2deg, abs, vstack, hstack, shape, eye, zeros, random, savez, load
from numpy import polyfit, poly1d, count_nonzero

from numpy import float64, average,matmul,dot

from numpy.linalg import norm, det,pinv
import multiprocessing as mp
from math import atan2, pi, asin, acos
from geometry_msgs.msg import Pose, Point, Quaternion

from numpy import sum,matmul
############# ROS Dependencies #####################################
import rospy
import os
from geometry_msgs import msg
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped, WrenchStamped, PointStamped
from std_msgs.msg import Bool, Float32,Int16,String,MultiArrayDimension,MultiArrayLayout

from std_msgs.msg import Header,Float64

from sensor_msgs.msg import Joy, JointState, PointCloud
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial import distance as dist_scipy
from numpy import sum,average,mean,ones,dot,multiply
from tf.transformations import quaternion_matrix
#from pytransform3d.urdf import UrdfTransformManager
from ipaddress import collapse_addresses
from itertools import chain
import queue
from re import T
from numpy.core.numeric import cross
from geometry_msgs import msg
import rospy
from numpy import matrix, matmul, transpose, isclose, array, rad2deg, abs, vstack, hstack, shape, eye, zeros

from threading import Thread, Lock

import threading

import multiprocessing

from example_robot_data import load



import scipy.optimize as sco
## Mutex operator
from threading import Lock
import time
import threading
## Multiprocessing toolbox for Jacobian function

from multiprocessing import Process

# Import services here

from rospygradientpolytope.srv import IKopt, IKoptResponse


from tf.transformations import quaternion_matrix


mutex = Lock()




from numpy.linalg import norm, det, pinv, solve
from math import atan2, pi, asin, acos
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped, WrenchStamped, PointStamped
from std_msgs.msg import Bool,Float64
from sensor_msgs.msg import Joy, JointState, PointCloud,Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from copy import copy

import tf_conversions as tf_c
from tf2_ros import TransformBroadcaster

from kuka_rsi_hw_interface.srv import *



# Polygon plot for ROS - Geometry message
from jsk_recognition_msgs.msg import PolygonArray, SegmentArray
from geometry_msgs.msg import Polygon, PolygonStamped, Point32


## Do not distribute this library
from rospygradientpolytope.linearalgebra import proj_point_plane,V_unit


# Old library components - Refer to catkin_telebot_ws for the Rviz plane messages
#from rospygradientpolytope.visual_polytope import velocity_polytope, desired_polytope, velocity_polytope_with_estimation
#from rospygradientpolytope.polytope_ros_message import create_plane_msg,create_polytopes_msg, create_polygon_msg, create_capacity_vertex_msg, create_segment_msg



## Do not distribute this library
from rospygradientpolytope.visual_polytope import velocity_polytope, desired_polytope, velocity_polytope_with_estimation,cartesian_velocity_polytope
from rospygradientpolytope.visual_polytope import cartesian_cmp_polytope, cartesian_velocity_with_joint_limit_polytope, cartesian_cmp_hsm_polytope
from rospygradientpolytope.polytope_ros_message import create_polytopes_msg,create_fast_polytopes_msg, create_polygon_msg, create_capacity_vertex_msg, create_segment_msg
from rospygradientpolytope.polytope_functions import get_polytope_hyperplane, get_capacity_margin, get_constraint_obstacle_jacobian
#from rospygradientpolytope.polytope_gradient_functions_optimized import Gamma_hat_gradient
from rospygradientpolytope.polytope_gradient_functions import Gamma_hat_gradient,Gamma_hat_gradient_dq

from rospygradientpolytope.sawyer_functions import jacobianE0, position_70
from rospygradientpolytope.robot_functions import getHessian, getJ_pinv
from rospygradientpolytope.linearalgebra import check_ndarray

import PyKDL

from urdf_parser_py.urdf import URDF

# from kdl_parser_py import KDL
from kdl_parser_py import urdf

#import open3d as o3d
### For service - Fixture line detection
from std_srvs.srv import Trigger, TriggerRequest

from std_msgs.msg import Float64MultiArray,Int32

from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.joint_kinematics import JointKinematics

from example_robot_data import load


### Import Pinnochio for Kinematics and Dynamics instead of KDL here
import pinocchio as pin


import time

from os.path import dirname, join, abspath






# getting the node namespace
namespace = rospy.get_namespace()

# For joint angle in URDF to screw



mutex = Lock()

mutex3 = Lock()

eps    = 1e-5
IT_MAX = 1000
DT     = 1e-1
damp   = 1e-12




# Load the urdf model
urdf_model_filename = '/home/imr/.local/lib/python3.8/site-packages/cmeel.prefix/share/example-robot-data/robots/dual_robot_description/urdf/kuka_meca.urdf'






robot_suffix = "_dualarm"




# import pycapacity 
import pycapacity as pycap

# get panda robot usinf example_robot_data
robot = load('kuka_meca')

# Load the urdf model
#robot.model = pin.buildModelFromUrdf(urdf_model_filename)

# get joint position ranges
q_max = robot.model.upperPositionLimit.T
q_min = robot.model.lowerPositionLimit.T
q_mean = (q_max+q_min)/2.0
print('q_mean',q_mean)
#input('stop q-mean')
# get max velocity
dq_max = robot.model.velocityLimit
dq_min = -dq_max

# Use robot configuration
# q0 = np.random.uniform(q_min,q_max)
q0 = (q_min+q_max)/2



# calculate the jacobian
data = robot.model.createData()

pin.framesForwardKinematics(robot.model,data,q0)
pin.computeJointJacobians(robot.model,data, q0)

# end-effector pose
Xee = data.oMf[robot.model.getFrameId(robot.model.frames[-1].name)]


urdf_model_path = '/home/imr/.local/lib/python3.8/site-packages/cmeel.prefix/share/example-robot-data/robots/dual_robot_description/urdf/kuka_meca.urdf'
mesh_dir = '/home/imr/.local/lib/python3.8/site-packages/cmeel.prefix/share/example-robot-data/robots/dual_robot_description/meshes/'
#geom_model = pin.buildGeomFromUrdf(robot.model,urdf_model_path,mesh_dir,pin.GeometryType.COLLISION)



geom_model = robot.collision_model

geom_data = pin.GeometryData(geom_model)
# geom_data.collisionRequest.enable_contact=True

print('geom_data',geom_data)

# Compute all the collisions
pin.computeCollisions(robot.model,data,geom_model,geom_data,q0,False)





a = pin.computeDistances(robot.model,data,geom_model,geom_data,q0)
b = geom_data.distanceResults[0]

#print('a result',a)
#print('b result',b)
geom_model = robot.collision_model
geom_data = pin.GeometryData(geom_model)
# Compute all the collisions
 

 
# Compute for a single pair of collision
pin.updateGeometryPlacements(robot.model,data,robot.collision_model,geom_data,q0)



J = pin.getFrameJacobian(robot.model, data, robot.model.getFrameId(robot.model.frames[-1].name), pin.LOCAL_WORLD_ALIGNED)
# use only position jacobian
J = J[:3,:]

# end-effector pose
Xee = data.oMf[robot.model.getFrameId(robot.model.frames[-1].name)]

# ## visualise the robot
from pinocchio.visualize import MeshcatVisualizer

viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
# # Start a new MeshCat server and client.
viz.initViewer(open=True)
# Load the robot in the viewer.
viz.loadViewerModel()
viz.display(q0)


class Geomagic2KUKA():
    def __init__(self):
        rospy.init_node('Geo2KUKA', anonymous=True)

        self.rmodel = robot.model
        self.rdata = data
        self.geom_data = geom_data
        self.geom_model = geom_model
        self.no_of_joints = self.rmodel.njoints -1 ## Not including a fixed frame

        self.active_joints = 6



        # get joint position ranges
        self.q_max = robot.model.upperPositionLimit.T
        self.q_min = robot.model.lowerPositionLimit.T
        self.q_mean = (self.q_max+self.q_min)/2.0

        print('self.q_mean is',self.q_mean)
        #input('stop q-mean')
        self.psi_max = 1.0*zeros(self.no_of_joints)
        self.psi_min = 1.0*zeros(self.no_of_joints)

        self.qdot_max = dq_max
        self.qdot_min = -1.0*self.qdot_max


        self.q_upper_limit = self.q_max
        self.q_lower_limit = self.q_min

        self.damp = 1e-12
        self.DT = 0.004





        ## Structure of the addFrame is as follows:
        ## input args: 'string' - frame_name, 'parent Joint id', 'bool'- if you want to carry inertia
        # SE3- placement of the frame, # type of frame - OP_FRAME meaning operational frame

        

        self.p_Hrep_A = Float64MultiArray()
        self.p_Hrep_b = Float64MultiArray()

        self.p_Hrep_size = Float64MultiArray()
        self.p_Hrep_size_dim = MultiArrayLayout()

        self.publish_velocity_polytope = rospy.Publisher(
            "/available_velocity_polytope"+robot_suffix, PolygonArray, queue_size=100)
        

        # Polytope in its H-rep from the polytope module
        '''
        self.publish_velocity_polytope_Hrep_A = rospy.Publisher(
            "/available_velocity_polytope_p_H_A"+robot_suffix, Float64MultiArray, queue_size=100)
        
        self.publish_velocity_polytope_Hrep_b = rospy.Publisher(
            "/available_velocity_polytope_p_H_b"+robot_suffix, Float64MultiArray, queue_size=100)
        
        self.publish_velocity_polytope_Hrep_size = rospy.Publisher(
            "/available_velocity_polytope_p_H_size"+robot_suffix, Float64MultiArray, queue_size=100)

        
        self.publish_desired_polytope = rospy.Publisher(
            "/desired_velocity_polytope"+robot_suffix, PolygonArray, queue_size=100)
        self.publish_capacity_margin_polytope = rospy.Publisher(
            "/capacity_margin_polytope"+robot_suffix, PolygonArray, queue_size=100)
        self.publish_vertex_capacity = rospy.Publisher(
            "/capacity_margin_vertex"+robot_suffix, PointStamped, queue_size=1)
        self.publish_vertex_proj_capacity = rospy.Publisher(
            "/capacity_margin_proj_vertex"+robot_suffix, PointStamped, queue_size=1)
        self.publish_capacity_margin_actual = rospy.Publisher(
            "/capacity_margin_actual"+robot_suffix, SegmentArray, queue_size=1)
        '''
        # publish plytope --- Estimated Polytope - Publisher
        '''
        self.publish_velocity_polytope_est = rospy.Publisher(
            "/available_velocity_polytope_est"+robot_suffix, PolygonArray, queue_size=100)
        self.publish_capacity_margin_polytope_est = rospy.Publisher(
            "/capacity_margin_polytope_est"+robot_suffix, PolygonArray, queue_size=100)
        self.publish_vertex_proj_capacity_est = rospy.Publisher(
            "/capacity_margin_proj_vertex_est"+robot_suffix, PointStamped, queue_size=1)
        self.publish_capacity_margin_actual_est = rospy.Publisher(
            "/capacity_margin_actual_est"+robot_suffix, SegmentArray, queue_size=1)

        self.publish_vertex_pose = rospy.Publisher(
            "/ef_pose_vertex"+robot_suffix, PointStamped, queue_size=1)
        self.publish_vertex_desired_pose = rospy.Publisher(
            "/ef_desired_pose_vertex"+robot_suffix, PointStamped, queue_size=1)
        '''
        self.polytope_display = False

        self.polytope_display_on_sub = rospy.Subscriber("polytope_show"+robot_suffix,Bool,self.polytope_show_on_callback)
        #self.start_interactive_ik_sub = rospy.Subscriber("run_ik",Bool,self.start_interactive_ik)
        #self.pub_end_ik = rospy.Publisher("ik_progress",Int16,queue_size=1)
        #self.pub_status_ik = rospy.Publisher("status_ik",String,queue_size=1)
        #self.sub_ik_pos = rospy.Subscriber("interactive_sphere",Pose,self.ik_pose_callback)

       
        self.q_in_numpy = zeros(self.no_of_joints)

        self.qdot_out = zeros(self.no_of_joints)
        self.qdot_out_meca = zeros(self.no_of_joints)
        




        self.geomagic_subscriber = rospy.Subscriber(
            "geomagic_twist", TwistStamped, self.geo_to_robot_callback, queue_size=1)

        
        self.polytope_plot_subscriber = rospy.Subscriber(
            "buttons_geo", Joy, self.plot_polytope, queue_size=1)
        
        self.EnterFlag = True

        self.kuka_joint_states_subscriber = rospy.Subscriber(
            "/joint_states", JointState, self.kuka_callback, queue_size=1)

        self.meca_joint_states_subscriber = rospy.Subscriber("/MecademicRobot_joint_fb",JointState, self.meca_callback, queue_size=1)
        self.kuka_joint_states_publisher = rospy.Publisher(
            "/position_trajectory_controller/command", JointTrajectory, queue_size=1)
        
        

        self.button_robot_subscriber = rospy.Subscriber(
            "buttons_geo", Joy, self.button_robot_update_callback, queue_size=1)
        
        self.collider_state_subscriber = rospy.Subscriber(
            "object_collider", Bool, self.collision_state_callback, queue_size=1)

        # self.meca_pose_publisher = rospy.Publisher("MecademicRobot_pose", Pose, queue_size=1)
        self.kuka_joints_publisher = rospy.Publisher(
            "KUKA_vel", TwistStamped, queue_size=1)
        # self.kuka_gripper_state_subscriber = rospy.Subscriber(
        #     "kuka_gripper_state_topic", Bool, self.gripper_actuate_callback, queue_size=1)
        
        self.robot_gripper_actuate = rospy.Subscriber(
            "gripper_state_topic", Bool, self.gripper_actuate_callback, queue_size=1)
        # self.meca_gripper_close = rospy.Subscriber(
        #     "kuka_gripper_close_topic", Bool, self.gripper_close_callback, queue_size=1)
        
        
        self.kuka_gripper_current_state = rospy.Publisher(
            "kuka_gripper_current_state", Bool, queue_size=1)
        
        self.meca_gripper_state_publisher = rospy.Publisher("MecademicRobot_gripper", Bool, queue_size=1)

        
        self.force_torque_ati_kuka = rospy.Subscriber(
            "/ati_force_torque_sensor_2/transformed_world", WrenchStamped, self.ft_kuka_callback, queue_size=1)
        

        self.force_torque_ati_meca = rospy.Subscriber(
            "/ati_force_torque_sensor_1/netft_data", WrenchStamped, self.ft_meca_callback, queue_size=1)


        


        # Rosbag - start service - Initializing parameter to call the service
        self.start_rosbag = rospy.Subscriber("/rosbag_start_topic", Bool, self.rosbag_start_record_callback,queue_size=1)

        self.stop_rosbag = rospy.Subscriber("/rosbag_stop_topic", Bool, self.rosbag_stop_record_callback,queue_size=1)




        self.end_effector_pos = rospy.Publisher("/end_effector_pos",PoseStamped,queue_size=1)
        self.end_effector_position = rospy.Publisher("/end_effector_position",PointStamped,queue_size=1)

        self.end_effector_position_kuka = rospy.Publisher("/grasp_frame_kuka",PointStamped,queue_size=1)
        self.end_effector_position_meca = rospy.Publisher("/grasp_frame_meca",PointStamped,queue_size=1)


        self.meca_joint_states_publisher   = rospy.Publisher("MecademicRobot_joint", JointState, queue_size=1)

        # self.meca_gripper_open = rospy.Subscriber(
        #     "gripper_state_topic", Bool, self.meca_gripper_actuate_callback, queue_size=1)


        

        #self.fixture_plane_distance = rospy.Publisher('/fixture_plane_distances', Float64, latch=True, queue_size=3)
        # self.meca_twist_publisher = rospy.Publisher("twist_test", Twist, queue_size=1)
        self.scalerXYZ = [0.5, 0.5, 0.5]
        self.mecaOffsetXYZ = [0.160, 0.0, 0.225]

        self.geomagic_offset = [0.1314, -0.16, 0.1]
        self.haptic_twist = TwistStamped()
        self.robot_joint_states = JointState()
        self.robot_joint_states.position = zeros(self.no_of_joints-1,dtype=float)

        self.kuka_joint_states = JointState()
        self.kuka_joint_states.position = zeros(6,dtype=float)    # [0.0,0.0,0.0,0.0,0.0,0.0]

        self.meca_joint_states = JointState()
        self.meca_joint_states.position = zeros(6,dtype=float)


        self.start_linearvelocity_state = False
        self.start_angularvelocity_state = False
        self.geo_pose_orientation_prev = array([0.0, 0.0, 0.0])

        self.pub_rate = 500  # Hz
        self.fixed_magnitude = 30
        #self.vel_scale = 3.5*self.fixed_magnitude*array([1.0, 1.0, 1.0])

        #self.vel_scale =1.0*self.fixed_magnitude*array([1.0, 1.0, 1.0])
        self.vel_scale =0.2*self.fixed_magnitude*array([1.0, 1.0, 1.0])

        
        self.angular_vel_scale = 10.0*array([1.0, 1.0, 1.0])
        
        self.angular_velocity_vector = matrix([[0.0], [0.0], [0.0]])
        self.previous_msg_state = False
        self.previous_msg = matrix([[0, 0, 0, 0, 0, 0]])
        self.button_robot_state = [0, 0]
        self.change_gripper_state = False
        self.changing_state = False

        self.current_collision_state = True
        self.angle_rot_previous_z = 0.0
        self.angle_rot_z = 0.0
        self.angle_rot_previous_y = 0.0
        self.angle_rot_y = 0.0
        self.angle_rot_previous_x = 0.0
        self.angle_rot_x = 0.0

        # Gripper current state
        self.gripper_closed = False
        self.ang_vel_x_prev = 0
        self.ang_vel_y_prev = 0
        self.ang_vel_z_prev = 0

        self.flag_linear = False
        self.flag_angular = False

        self.lin_vel_x_prev = 0
        self.lin_vel_y_prev = 0
        self.lin_vel_z_prev = 0


        # Collision zone flags here
        self.collision_zone_counter = 0
        self.leave_flag_collision_zone = False

        # Cartesian Wrench is declared here
        # self.cartesian_wrench = WrenchStamped()

        # ft_sensor - for KUKA data is declared here
        self.ft_sensor_kuka = WrenchStamped()
        self.ft_sensor_meca = WrenchStamped()

        self.base_line_counter = 0
        

        
        ## I dont know wtf we do the log to get to twist 
        ## Need to read about Lie Algebra
        
        self.cartesian_twist = pin.log(pin.SE3.Identity()).vector

        

        # Limits of all jointts are here

        self.robot_joint_names = ['joint_a1', 'joint_a2',
            'joint_a3', 'joint_a4', 'joint_a5', 'joint_a6','meca_axis_1_joint','meca_axis_2_joint','meca_axis_3_joint','meca_axis_4_joint'\
                                  ,'meca_axis_5_joint','meca_axis_6_joint']

        self.robot_joint_names_pub = self.robot_joint_names

        # self.q_upper_limit = [
        #     robot_description.joint_map[i].limit.upper - 0.07 for i in self.robot_joint_names]
        # self.q_lower_limit = [
        #     robot_description.joint_map[i].limit.lower + 0.07 for i in self.robot_joint_names]

        # self.qdot_limit = [
        #     robot_description.joint_map[i].limit.velocity for i in self.robot_joint_names]

        self.gripper_state_msg = Bool()
        # self.gripper_state_msg.data = False

        ########################
        ## Get frame transformation here - End effector frame with respect to base
        ## Frame - frame.p = position 3x1 vector, frame.M = Rotational matrix of the frame

        '''
        self.eeFrame = kdl_chain.getSegment(0).getFrameToTip()

        self.baseFrame = PyKDL.Frame.Identity()

        self.cam_rot = PyKDL.Rotation()
        self.cam_rot = self.cam_rot.RPY(0,0,0)
        '''
        #self.cam_rot = PyKDL.Rotation()
        #self.cam_rot = self.cam_rot.RPY(0,0,0)

        

        self.plane_verts = []
        self.pose_verts = []

        ## Distance between virtual guide planes
        self.dist_plane = 0.005 # 3 mm 
        self.dist_tol = 0.002

        self.qdot_limit = self.qdot_max

        # self.qdot_limit = [
        #     robot_urdf.joint_map[i].limit.velocity for i in self.robot_joint_names]

        # self.qdot_max = array(self.qdot_limit)
        # self.qdot_min = -1*self.qdot_max
        self.fun_iter = Int16()
        self.fun_iter.data = 0
        self.start_optimization_bool = False

        self.msg_status_ik = String()

        # print('self.qdot_max', self.qdot_max)
        # print('self.qdot_min', self.qdot_min)
        #self.q_in = zeros(6)


        self.plot_polytope_thread = None
        self.thread_is_running = False
        #self.thread_is_running = True

        #self.thread_cm_is_running = False

        self.force_baseline_arr_meca = zeros(shape=(500,3))
        self.torque_baseline_arr_meca = zeros(shape=(500,3))

        self.force_baseline_meca = zeros(shape=(3))
        self.torque_baseline_meca = zeros(shape=(3))
        self.baseline_record_once = True


        self.force_norm = 0
        self.torque_norm = 0


        ## Hard-coded value

        self.obstacle_link_vector = zeros(shape = (12,3))
        self.obstacle_dist_vector = zeros(shape = (12))
        self.scaled_maximum_vector = ones(shape=(12))
        self.danger_threshold = 0.1

        self.polytope_verts_cmp = array([])
        self.polytope_faces_cmp = array([])

        







        #self.q_bounds = zeros(len(self.q_upper_limit),2)

        

        
        self.fun_counter = 0

        self.color_array_cm = ['g','r']
        
        self.time_arr = zeros(shape=(2))

        ############3 Python Attributes ####################################

        # self.joints_name = list(tm._joints)

        pin.forwardKinematics(self.rmodel,self.rdata, self.q_in_numpy)
        pin.updateFramePlacements(self.rmodel,self.rdata)
        # self.grasp_frame_r1_SE3 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')]
        # print('self.rdata.oMf[self.rmodel.getFrameId_tcp_kuka',self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')])
        # self.grasp_frame_r1_SE3.translation = (self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation - self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation)*0.5
        # print('self.grasp_frame_r1_SE3.translation',self.grasp_frame_r1_SE3.translation)
        # self.grasp_frame_r2_SE3 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')]
        # self.grasp_frame_r2_SE3.translation = (self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation - self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation)*0.5

        # print('self.grasp_frame_r2_SE3.translation',self.grasp_frame_r2_SE3.translation)
        
        # #self.rmodel.getFrameId('tcp_kuka')
        # self.grasp_frame_kuka_id = self.rmodel.addFrame(pin.Frame('grasp_frame_r1',6,self.rmodel.getFrameId('tcp_kuka'),self.grasp_frame_r1_SE3,pin.FrameType.OP_FRAME)) # Returns the ID of the frame
        # self.grasp_frame_meca_id= self.rmodel.addFrame(pin.Frame('grasp_frame_r2',12,self.rmodel.getFrameId('tcp_meca'),self.grasp_frame_r2_SE3,pin.FrameType.OP_FRAME)) # Returns the ID of the frame
                
        # frame_info = self.rmodel.frames[self.grasp_frame_kuka_id]
        # print(frame_info.placement.translation)
        

        #input('stop here')

        self.publish_velocity_polytope = rospy.Publisher(
            "/available_velocity_polytope"+robot_suffix, PolygonArray, queue_size=100)
        
        self.publish_velocity_cmp = rospy.Publisher(
            "/available_cmp_polytope"+robot_suffix, PolygonArray, queue_size=100)
        

        self.publish_velocity_cmp_hsm = rospy.Publisher(
            "/available_cmp_hsm_polytope"+robot_suffix, PolygonArray, queue_size=100)
        

        self.cp1_publisher_msg = rospy.Publisher(
            "/cp1_point"+robot_suffix, PointStamped, queue_size=100)
        
        self.cp2_publisher_msg = rospy.Publisher(
            "/cp2_point"+robot_suffix, PointStamped, queue_size=100)
        
        self.publish_velocity_polytope = rospy.Publisher(
            "/available_velocity_polytope", PolygonArray, queue_size=100)
        self.publish_desired_polytope = rospy.Publisher(
            "/desired_velocity_polytope", PolygonArray, queue_size=100)
        self.publish_capacity_margin_polytope = rospy.Publisher(
            "/capacity_margin_polytope", PolygonArray, queue_size=100)
        self.publish_vertex_capacity = rospy.Publisher(
            "/capacity_margin_vertex", PointStamped, queue_size=1)
        self.publish_vertex_proj_capacity = rospy.Publisher(
            "/capacity_margin_proj_vertex", PointStamped, queue_size=1)
        self.publish_capacity_margin_actual = rospy.Publisher(
            "/capacity_margin_actual", SegmentArray, queue_size=1)

        # publish plytope --- Estimated Polytope - Publisher

        self.publish_velocity_polytope_est = rospy.Publisher(
            "/available_velocity_polytope_est", PolygonArray, queue_size=100)
        self.publish_capacity_margin_polytope_est = rospy.Publisher(
            "/capacity_margin_polytope_est", PolygonArray, queue_size=100)
        self.publish_vertex_proj_capacity_est = rospy.Publisher(
            "/capacity_margin_proj_vertex_est", PointStamped, queue_size=1)
        self.publish_capacity_margin_actual_est = rospy.Publisher(
            "/capacity_margin_actual_est", SegmentArray, queue_size=1)

        self.publish_vertex_pose = rospy.Publisher(
            "/ef_pose_vertex", PointStamped, queue_size=1)
        self.publish_vertex_desired_pose = rospy.Publisher(
            "/ef_desired_pose_vertex", PointStamped, queue_size=1)

        # Subscribe joints of Robot --- Joint State subscriber

        self.robot_joint_state_publisher = rospy.Publisher(
            "/joint_states", JointState, queue_size=1)
                
        self.polytope_display = False
        
        self.polytope_display_on_sub = rospy.Subscriber("polytope_show",Bool,self.polytope_show_on_callback)
        self.start_interactive_ik_sub = rospy.Subscriber("run_ik",Bool,self.start_interactive_ik)
        self.pub_end_ik = rospy.Publisher("ik_progress",Int16,queue_size=1)
        self.pub_status_ik = rospy.Publisher("status_ik",String,queue_size=1)
        self.sub_ik_pos = rospy.Subscriber("interactive_sphere",Pose,self.ik_pose_callback)


        #self.robot_joint_state_subscriber = rospy.Subscriber("/joint_states",JointState,self.joint_state_callback,queue_size=1)

        # Paper cartesian desired polytope
        self.cartesian_desired_vertices = 1.0*array([[0.20000, 0.50000, 0.50000],
                                                     [0.50000, -0.10000, 0.50000],
                                                     [0.50000, 0.50000, -0.60000],
                                                     [0.50000, -0.10000, -0.60000],
                                                     [-0.30000, 0.50000, 0.50000],
                                                     [-0.30000, -0.10000, 0.50000],
                                                     [-0.30000, 0.50000, -0.60000],
                                                     [-0.30000, -0.10000, -0.60000]])

        self.cartesian_desired_vertices = 0.05*array([[0.20000, 0.50000, 0.50000],
                                                     [0.50000, -0.10000, 0.50000],
                                                     [0.50000, 0.50000, -0.60000],
                                                     [0.50000, -0.10000, -0.60000],
                                                     [-0.30000, 0.50000, 0.50000],
                                                     [-0.30000, -0.10000, 0.50000],
                                                     [-0.30000, 0.50000, -0.60000],
                                                     [-0.30000, -0.10000, -0.60000]])
        

        # Create an interactive marker server
        # Create an interactive marker server

        
        
        self.desired_pose = Pose()


        self.desired_vertices = zeros(
            shape=(len(self.cartesian_desired_vertices), 3))

        self.desired_vertices = self.cartesian_desired_vertices

        self.pub_rate = 500  # Hz

        self.sigmoid_slope = 150

        self.sigmoid_slope_input = 5

        self.fun_iter = Int16()
        self.fun_iter.data = 0
        self.start_optimization_bool = False

        self.msg_status_ik = String()

        print('self.qdot_max', self.qdot_max)
        print('self.qdot_min', self.qdot_min)
        self.q_in = zeros(12)

        jac_output = mp.Array("f",[0,0,0,0,0,0,0,0,0,0,0,0])

        self.plot_polytope_thread = None
        self.thread_is_running = False
        #self.thread_cm_is_running = False


        #self.q_test = zeros(7)

        # self.pykdl_util_kin = KDLKinematics(
        #     robot_urdf, base_link, tip_link, None)
        #self.q_bounds = zeros(len(self.q_upper_limit),2)

        

        self.q_upper_limit = array([self.q_max]).T
        #self.q_upper_limit = self.pykdl_util_kin.joint_limits_upper
        self.q_lower_limit = array([self.q_min]).T
        #self.q_lower_limit = self.pykdl_util_kin.joint_limits_lower

        self.q_bounds = hstack((self.q_lower_limit, self.q_upper_limit))
        sigmoid_slope_test = array([50, 100, 150, 200, 400])

        self.sigmoid_slope_array = array([50, 100, 150, 200, 400])

        self.cm_est = None

        self.time_counter = 0
        self.fun_counter = 0

        self.color_array_cm = ['g','r']
        self.cm_est_arr = zeros(shape=(2))
        self.cm_est_arr[:] = -10000
        self.time_arr = zeros(shape=(2))
    
        #self.plot_polytope()
    

    def ik_pose_callback(self,desired_ik_pose):
        print('this is what i am ')
        self.desired_ik_pose = desired_ik_pose.position
        print('desired ik pose is',self.desired_ik_pose)
    def start_interactive_ik(self,start_optimization):
        print('start_IK',start_optimization.data)
        self.start_optimization_bool = start_optimization.data
        if self.start_optimization_bool:
            print('Start IK')            
            self.fun_iter.data = 0
            self.fun_counter = 0
            self.compute_pose_ik(pos_ik=array([self.desired_ik_pose.x,self.desired_ik_pose.y,self.desired_ik_pose.z]))  # Picture - FEasible pose - 1 - Good
    def start_plot_thread(self):
        if self.thread_is_running:
            print("Thread already running!")
            return
        self.plot_polytope_thread = threading.Thread(target=self.plot_polytope)
        self.thread_is_running = True
        self.plot_polytope_thread.start()

        #input('I have started thread')
    def stop_thread(self):
        self.thread_is_running = False
        print('Stopping thread')
    '''
    def start_cm_plot_thread(self):
        if self.thread_cm_is_running:
            print("Capacity PLotting Thread already running!")
            return
        self.plot_cm_thread = threading.Thread(target=self.plot_capacity_margin_est)
        self.thread_cm_is_running = True
        self.plot_cm_thread.start()

        #input('I have started thread')
    def stop_cm_thread(self):
        self.thread_cm_is_running = False
        print('Stopping CM Plot thread')
    '''
    def processfeedback(self, feedback):
        self.desired_pose.position.x = feedback.pose.position.x
        self.desired_pose.position.y = feedback.pose.position.y
        self.desired_pose.position.z = feedback.pose.position.z

    def polytope_show_on_callback(self,show_bool):
        self.polytope_display = show_bool.data

        if self.polytope_display:
            self.start_plot_thread()
            #self.start_cm_plot_thread()
        else:
            self.stop_thread()
            #self.stop_cm_thread()

    def plot_capacity_margin_est(self,cm_est):

        '''
        while self.thread_cm_is_running:
            if self.polytope_display:

                #print('plotting here')
        '''
        self.cm_est = cm_est
        if self.cm_est != None:
            
            if self.cm_est > 0:
                color_arr_cm = 'g'
            else:
                color_arr_cm = 'r'                
        

            if self.cm_est_arr[0] != -10000 and self.cm_est_arr[1] == -10000:

                self.cm_est_arr[1] = self.cm_est
                self.ax_cm.scatter(self.time_counter,self.cm_est,color=color_arr_cm)
                
            if self.cm_est_arr[0] == -10000:
                self.cm_est_arr[0] = self.cm_est     
            
                self.ax_cm.scatter(self.time_counter,self.cm_est,color=color_arr_cm)

            else:
                x = [self.time_counter-1,self.time_counter]
                
                self.cm_est_arr[0] =  self.cm_est_arr[1]
                self.cm_est_arr[1] =  self.cm_est
                self.ax_cm.plot(x,self.cm_est_arr,color=color_arr_cm)
                #self.cm_est_arr



            


            #self.fig_cm.canvas.flush_events()
            self.time_counter += 1

            # drawing updated values
            if self.polytope_display:
                self.fig_cm.canvas.draw()
            else:
                for artist in plt.gca().lines + plt.gca().collections:
                    artist.remove()
                    self.time_counter = 0



    def compute_pose_ik(self, pos_ik):
        import time
        from numpy.linalg import det
        from numpy import sum, mean, average, linspace
        import matplotlib.pyplot as plt

        
        q0 = zeros(self.active_joints)
        for j in range(6,self.no_of_joints):
            q0[6-j] = random.uniform(
                self.q_lower_limit[6-j], self.q_upper_limit[6-j])
        

        #for i in range(0, 1000):

        st = time.time()
        using_pinnochio = False

        if using_pinnochio:
            q_opt = self.fmin_opt_ik(q0, pos_ik, True)
            q0 = q_opt
        else:
            q_opt = self.fmin_opt_ik(q0, pos_ik, True)
            print('q_opt is',q_opt)
            q_opt = self.fmin_opt(q_opt[6:], pos_ik, True)
            q0 = q_opt.x

        # To publish the joint states to Robot
        #self.joint_state_publisher_robot(q_opt.x)

        ex_time = time.time() - st
        print('execution time is',ex_time)




        
        #print('self.polytope_display',self.polytope_display)
    def joint_state_callback(self, robot_joints):

        # Get Joint angles of the Sawyer Robot
        # Interchanged joint positions here
        ## Be careful for UR robot the index is changed 0 and 2 are interchanged

        for i in range(self.no_of_joints):
            self.q_in[i] = robot_joints.position[i]

    # def check_gradient(self,q_in,step_size:int):
    def joint_state_publisher_robot(self, q_joints):

        q_in = q_joints
        #print('q_in joints are',q_in)
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.velocity = []
        msg.effort = []

        for i in range(self.no_of_joints):
            msg.name.append(self.robot_joint_names_pub[i])
            msg.position.append(q_in[i])
        #msg.position = [q_in[6],q_in[7],q_in[8],q_in[9],q_in[10],q_in[11],q_in[0],q_in[1],q_in[2],q_in[3],q_in[4],q_in[5]]
            
        self.robot_joint_state_publisher.publish(msg)

        '''
        if self.polytope_display:
            self.plot_polytope_thread = mp.Process(target=self.plot_polytope,args=(q_joints))
            self.plot_polytope_thread.start()
            #self.plot_polytope_thread.join()
        '''
        self.q_in = q_in
        

    def plot_polytope(self):                                                                                                                                                                                                                 
        
        while self.thread_is_running:
            if self.polytope_display:

                
                #print('I am plotting')
                viz.display(self.q_in_numpy)


                mutex.acquire()
                pin.computeFrameJacobian(self.rmodel, self.rdata,self.q_in_numpy,33)
                pin.forwardKinematics(self.rmodel,self.rdata, self.q_in_numpy)
                pin.updateFramePlacements(self.rmodel,self.rdata)


                pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
                pos_act2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation



                pos_act = pos_act1 + (pos_act2 - pos_act1)*0.5

                
                pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

                
                J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_kuka'), pin.LOCAL_WORLD_ALIGNED)
                J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
                
                J_Hess = hstack((J_Hess1[:,:6],J_Hess2[:,6:]))

                # print('pos_act_mat',pos_act_mat)

                #pos_act = position_70(q_in)
                # print('self.pos_reference',self.pos_reference)
                #print('Current position in optimization is', pos_act)
                #input('Wait here ')
                # print('norm,',norm(pos_act-self.pos_reference))
                # print('self.pos_reference',self.pos_reference)
                #print('pos_act', pos_act)

                #print('distance error norm',distance_error)
                

                scaling_factor = 10.0
                
                ### Polytope plot with estimation
                
                #pykdl_kin_jac = pykdl_util_kin.jacobian(self.q_in_numpy)
                polytope_verts, polytope_faces, facet_vertex_idx, capacity_faces, capacity_margin_proj_vertex, \
                    polytope_verts_est, polytope_faces_est, capacity_faces_est, capacity_margin_proj_vertex_est,cm_index = \
                                                velocity_polytope_with_estimation(J_Hess,self.qdot_max,self.qdot_min,self.desired_vertices,self.sigmoid_slope_input)
                desired_polytope_verts, desired_polytope_faces = desired_polytope(self.desired_vertices)


                self.cm_est = cm_index
                #print('self.cm_est',self.cm_est)
                #print('facet_vertex_idx',facet_vertex_idx)
                #print('capacity_margin_proj_vertex',capacity_margin_proj_vertex)
                #print('capacity_margin_proj_vertex_est',capacity_margin_proj_vertex_est)
                

                # Only for visualization - Polytope at end-effector - No physical significance
                #ef_pose = pykdl_util_kin.forward(self.q_in_numpy)[:3,3]

                
                

                # Get end-effector of the robot here for the polytope offset

                #ef_pose = position_70(q_in)

                ef_pose = pos_act
                #ef_pose = ef_pose[:,0]
                #print('ef_pose is',ef_pose)
                #input('stop to test ef_pose')

                ########### Actual POlytope plot ###########################################################################
                # Publish polytope faces
                polyArray_message = self.publish_velocity_polytope.publish(create_polytopes_msg(polytope_verts, polytope_faces, \
                                                                                                    ef_pose,"base_link", scaling_factor))
                
                
                ### Desired polytope set - Publish

                DesiredpolyArray_message = self.publish_desired_polytope.publish(create_polytopes_msg(desired_polytope_verts, desired_polytope_faces, \
                                                                                                    ef_pose,"base_link", scaling_factor))


                ### Vertex for capacity margin on the Desired Polytope
                #print('facet_vertex_idx',facet_vertex_idx)
                closest_vertex = self.cartesian_desired_vertices[facet_vertex_idx[0,1]]
                #print('closest_vertex',closest_vertex)

                CapacityvertexArray_message = self.publish_vertex_capacity.publish(create_capacity_vertex_msg(closest_vertex, \
                                                                                            ef_pose, "base_link", scaling_factor))

                
                
                ### Vertex for capacity margin on the Available Polytope
                CapacityprojvertexArray_message = self.publish_vertex_proj_capacity.publish(create_capacity_vertex_msg(capacity_margin_proj_vertex, \
                                                                                            ef_pose, "base_link", scaling_factor))


                ### Vertex for capacity margin on the Available Polytope
                ActualposevertexArray_message = self.publish_vertex_pose.publish(create_capacity_vertex_msg(ef_pose, \
                                                                                            array([0,0,0]), "base_link", 1))
                
                ### Vertex for capacity margin on the Available Polytope
                '''
                DesiredposevertexArray_message = self.publish_vertex_desired_pose.publish(create_capacity_vertex_msg(self.pos_reference, \
                                                                                        array([0,0,0]), "base_link", 1))
                '''
                ### Plane for capacity margin 
                #print('capacity_faces',capacity_faces)

                ### Vertex for capacity margin on the Available Polytope
                CapacitymarginactualArray_message = self.publish_capacity_margin_actual.publish(create_segment_msg(closest_vertex, \
                                                    capacity_margin_proj_vertex,ef_pose, "base_link", scaling_factor))
                
                capacityArray_message = self.publish_capacity_margin_polytope.publish(create_polytopes_msg(polytope_verts, capacity_faces, \
                                                                                                    ef_pose,"base_link", scaling_factor))


                ########### Estimated Polytope plot ###########################################################################
                
                # Publish polytope faces
                EstpolyArray_message = self.publish_velocity_polytope_est.publish(create_polytopes_msg(polytope_verts_est, polytope_faces_est, \
                                                                                                    ef_pose,"base_link", scaling_factor))
                
                

                ### Vertex for capacity margin on the Available Polytope
                EstCapacityprojvertexArray_message = self.publish_vertex_proj_capacity_est.publish(create_capacity_vertex_msg(capacity_margin_proj_vertex_est, \
                                                                                            ef_pose, "base_link", scaling_factor))


                ### Vertex for capacity margin on the Available Polytope
                EstCapacitymarginactualArray_message = self.publish_capacity_margin_actual_est.publish(create_segment_msg(closest_vertex, \
                                                    capacity_margin_proj_vertex_est,ef_pose, "base_link", scaling_factor))
                

                EstcapacityArray_message = self.publish_capacity_margin_polytope_est.publish(create_polytopes_msg(polytope_verts_est, capacity_faces_est, \
                                                                                                    ef_pose,"base_link", scaling_factor))

                


                ### Vertex 
                
                ##############################################################################################################
                
                
                #print('facet_vertex_idx',facet_vertex_idx)
                
                mutex.release()

    def ft_kuka_callback(self, ft_data_kuka):

        self.ft_sensor_kuka = ft_data_kuka
        torque = array([ft_data_kuka.wrench.torque.x, ft_data_kuka.wrench.torque.y, ft_data_kuka.wrench.torque.z ])
        forces = array([ft_data_kuka.wrench.force.x, ft_data_kuka.wrench.force.y, ft_data_kuka.wrench.force.z])
        wrench_arr = hstack([[torque,forces]])
        #print('wrench_arr - kuka',wrench_arr)
        pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

                
        J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_kuka'), pin.LOCAL_WORLD_ALIGNED)
        force_polytope_verts = matmul(transpose(J_Hess1[0:3,:6]),transpose(wrench_arr))

    
    
    def collision_update_callback(self):
        
        #mutex.acquire()

        # # Compute for a single pair of collision        
        distance_results = self.geom_data.distanceResults
        #print('distanceresults',distance_results)
        # Print the status of collision for all collision pairs
        counter = 0

        for result in distance_results: 
            #cr = geom_data.collisionResults[k].closestPoints
            #dr_result = distance_results[k]
            #print('result is',dir(result))
            #cp = robot.collision_model.collisionPairs[k]
            cp1 = result.getNearestPoint1()
            cp2 = result.getNearestPoint2()
            print('closest points 1',result.getNearestPoint1())
            print('closest points 2',result.getNearestPoint2())


            # msg1.point.x = cp1[0]
            # msg1.point.y = cp1[1]
            # msg1.point.z = cp1[2]

            # msg2.point.x = cp2[0]
            # msg2.point.y = cp2[1]
            # msg2.point.z = cp2[2]



            if (counter < 12):

                self.obstacle_link_vector[counter,0] = cp2[0] - cp1[0]
                self.obstacle_link_vector[counter,1] = cp2[1] - cp1[1]
                self.obstacle_link_vector[counter,2] = cp2[2] - cp1[2]
                #self.cp1_publisher_msg[counter].publish(msg1)
                #self.cp2_publisher_msg[counter].publish(msg2)

            counter += 1
        
            #print(dir(result))
            # print('closest points',dr.nearest_points[1].transpose())
        #print('total pairs are',counter)
            #print("collision pair:",cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")
        
        # Print the status of collision for all collision pairs
        # for k in range(len(robot.collision_model.collisionPairs)): 
        #     # cr = geom_data.collisionResults[k]
            
        #     # if cr.isCollision() == True:
        #     #     print('In collision')
        #     aaa = pin.computeCollisions(self.rmodel,self.rdata,self.geom_model,self.geom_data,self.q_in_collision,True)
        #     a = pin.computeDistances(self.rmodel,self.rdata,self.geom_model,self.geom_data,self.q_in_collision)

        #     b = self.geom_data.distanceResults[0]
        #     print('b',b)
        #     # res = self.geom_data.collisionResults[0]
        #     #assert(res.isCollision())
        #     # contact = res.getContact(0)
        #     # print(self.geom_model.collisionPairs[0],contact.normal.T,contact.pos.T)
        #     # cp = robot.collision_model.collisionPairs[k]
        #     # print("collision pair:",cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")
        #mutex.release()
    


    def ft_meca_callback(self, ft_data_meca):
        

        # Measure the baseline value here for the FT sensor data
        # if (self.base_line_counter < 500):
        #     #for i in range(self.base_line_counter):
        #     self.force_baseline_arr_meca[self.base_line_counter,0] = ft_data_meca.wrench.force.x
        #     self.force_baseline_arr_meca[self.base_line_counter,1] = ft_data_meca.wrench.force.x
        #     self.force_baseline_arr_meca[self.base_line_counter,2] = ft_data_meca.wrench.force.x
            
        #     self.torque_baseline_arr_meca[self.base_line_counter,0] = ft_data_meca.wrench.torque.x
        #     self.torque_baseline_arr_meca[self.base_line_counter,1] = ft_data_meca.wrench.torque.x
        #     self.torque_baseline_arr_meca[self.base_line_counter,2] = ft_data_meca.wrench.torque.x

        #     print(' array is',self.force_baseline_arr_meca[self.base_line_counter])
        #     self.base_line_counter += 1
        if (self.base_line_counter< 500):
            #if (self.baseline_record_once):
            # print('forces arr',self.force_baseline_arr_meca)
            # self.force_baseline_meca[0] = mean(self.force_baseline_arr_meca[:,0])
            # self.force_baseline_meca[1] = mean(self.force_baseline_arr_meca[:,1])
            # self.force_baseline_meca[2] = mean(self.force_baseline_arr_meca[:,2])


            # self.torque_baseline_meca[0] = mean(self.torque_baseline_arr_meca[:,0])
            # self.torque_baseline_meca[1] =mean(self.torque_baseline_arr_meca[:,1])
            # self.torque_baseline_meca[2] = mean(self.torque_baseline_arr_meca[:,2])
            # print('baseline forces are',self.force_baseline_meca)
            self.force_baseline_meca[0] = ft_data_meca.wrench.force.x
            self.force_baseline_meca[1] = ft_data_meca.wrench.force.y
            self.force_baseline_meca[2] = ft_data_meca.wrench.force.z

            self.torque_baseline_meca[0] = ft_data_meca.wrench.torque.x
            self.torque_baseline_meca[1] = ft_data_meca.wrench.torque.y
            self.torque_baseline_meca[2] = ft_data_meca.wrench.torque.z
            self.base_line_counter += 1
            #print('baseline forces are',self.force_baseline_meca)
            #print('baseline torques are',self.torque_baseline_meca)
            #input('Enteredd this loop')
            #self.force_baseline_meca = False

            #self.force_baseline_meca[0]
        else:
            
            
            torque = array([ft_data_meca.wrench.torque.x, ft_data_meca.wrench.torque.y, ft_data_meca.wrench.torque.z ])
            
            forces = array([ft_data_meca.wrench.force.x, ft_data_meca.wrench.force.y, ft_data_meca.wrench.force.z])

            #print('forces are',forces)
            #print('baseline forces are',self.force_baseline_meca)

            #print('torques are',torque)
            #print('baseline torques are',self.torque_baseline_meca)
            self.force_norm = norm(self.force_baseline_meca - forces)
            self.torque_norm = norm(self.torque_baseline_meca - torque)
            #print('force norm is',force_norm)
            #print('torque norm is',torque_norm)
            

            wrench_arr = hstack([[torque,forces]])
            #print('wrench_arr - meca',wrench_arr)
            J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
            force_polytope_verts = matmul(transpose(J_Hess2[0:3,6:]),transpose(wrench_arr))

        
    def polytope_show_on_callback(self,show_bool):
        self.polytope_display = show_bool.data

        if self.polytope_display:
            self.start_plot_thread()
            #self.start_cm_plot_thread()
        else:
            self.stop_thread()
            #self.stop_cm_thread()
    def start_plot_thread(self):
        if self.thread_is_running:
            print("Thread already running!")
            return
        self.plot_polytope_thread = threading.Thread(target=self.plot_polytope)
        self.thread_is_running = True
        self.plot_polytope_thread.start()

        #input('I have started thread')
    def stop_thread(self):
        self.thread_is_running = False
        print('Stopping thread')
    
    #def plot_polytope(self,plot_polytope_geo):
    
    def plot_polytope(self):
        
        
        #while self.thread_is_running:
        #    if self.polytope_display:

                
        
        #input('cjeck joints input')
        #print('I am plotting')
                # # Compute for a single pair of collision
        #if plot_polytope_geo.buttons[1]:  
            # print('plotting ')          
        while not rospy.is_shutdown():
            pin.updateGeometryPlacements(self.rmodel,self.rdata,self.geom_model,self.geom_data,self.q_in_numpy)
            pin.computeDistances(self.rmodel, self.rdata, self.geom_model, self.geom_data, self.q_in_numpy)
            distance_results = self.geom_data.distanceResults

            counter = 0

            
            time_begin = rospy.Time.now()


            pin.computeFrameJacobian(self.rmodel, self.rdata,self.q_in_numpy,33)
            pin.forwardKinematics(self.rmodel,self.rdata, self.q_in_numpy)
            pin.updateFramePlacements(self.rmodel,self.rdata)


            pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
            pos_act2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation



            pos_act = pos_act1 + (pos_act2 - pos_act1)*0.5

            
            pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

            
            J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_kuka'), pin.LOCAL_WORLD_ALIGNED)
            J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
            
            J_Hess = hstack((J_Hess1[:,:6],J_Hess2[:,6:]))

            

            scaling_factor = 80.0
            



            


            ef_pose = transpose(pos_act)


            distance_results = self.geom_data.distanceResults

            # Print the status of collision for all collision pairs
            counter = 0

            msg1 = PointStamped()
            msg1.header = Header()
            msg1.header.frame_id = 'telebot_cell_base_link'
            msg2 = PointStamped()
            msg2.header = Header()
            msg2.header.frame_id = 'telebot_cell_base_link'


            

            for result in distance_results: 
                #cr = geom_data.collisionResults[k].closestPoints
                #dr_result = distance_results[k]
                #print('result is',dir(result))
                #cp = robot.collision_model.collisionPairs[k]

                
                cp1 = result.getNearestPoint1()
                cp2 = result.getNearestPoint2()
                #print('closest points 1',result.getNearestPoint1())
                #print('closest points 2',result.getNearestPoint2())





                if (counter < 12):
                    self.obstacle_dist_vector[counter] = result.min_distance
                    if result.min_distance <= self.danger_threshold:

                        self.obstacle_link_vector[counter,0] = cp2[0] - cp1[0]
                        self.obstacle_link_vector[counter,1] = cp2[1] - cp1[1]
                        self.obstacle_link_vector[counter,2] = cp2[2] - cp1[2]


                        self.scaled_maximum_vector[counter] = result.min_distance**2

                    
                    if counter == 0:

                        msg1.point.x = cp1[0]
                        msg1.point.y = cp1[1]
                        msg1.point.z = cp1[2]

                        msg2.point.x = cp2[0]
                        msg2.point.y = cp2[1]
                        msg2.point.z = cp2[2]
                        self.cp1_publisher_msg.publish(msg1)
                        self.cp2_publisher_msg.publish(msg2)

                counter += 1

            
            ########### Actual POlytope plot ###########################################################################
            # Publish polytope faces
            # polyArray_message = self.publish_velocity_polytope.publish(create_polytopes_msg(polytope_verts, polytope_faces, \
            #                                                                                     ef_pose,"telebot_cell_base_link", scaling_factor))
            

            
            #print('obstacle_link_global is',self.obstacle_link_vector)
            #print('self.obstacle_dist_vector',self.obstacle_dist_vector)
            #input('stop and test')
            '''
            polytope_verts_cmp, polytope_faces_cmp,polytope_center,polytope_center_max = cartesian_cmp_polytope(J_Hess,self.q_in_numpy,self.qdot_min,self.qdot_max,\
                                                                                        self.q_min,self.q_max,self.q_mean,self.psi_max, self.psi_min, \
                                                                                            self.obstacle_link_vector, \
                                                                                                self.obstacle_dist_vector,self.danger_threshold)
            

            if polytope_verts_cmp.any():
                self.polytope_verts_cmp = polytope_verts_cmp
                self.polytope_faces_cmp = polytope_faces_cmp

            '''

            polytope_verts_cmp, polytope_faces_cmp,polytope_center,polytope_center_max = cartesian_cmp_hsm_polytope(J_Hess,self.q_in_numpy,self.qdot_min,self.qdot_max,\
                                                                                        self.q_min,self.q_max,self.q_mean,self.psi_max, self.psi_min, \
                                                                                            self.obstacle_link_vector, \
                                                                                                self.obstacle_dist_vector,self.danger_threshold)
            

            if polytope_verts_cmp.any():
                self.polytope_verts_cmp = polytope_verts_cmp
                self.polytope_faces_cmp = polytope_faces_cmp



            # polytope_verts_jpl, polytope_faces_jpl = cartesian_velocity_with_joint_limit_polytope(J_Hess,self.q_in_numpy,self.qdot_min,self.qdot_max,\
            #                                                                             self.q_min,self.q_max,self.q_mean,self.psi_max, self.psi_min)
            

            #J_coll = get_constraint_obstacle_jacobian(J_Hess,12,self.obstacle_link_vector,self.obstacle_dist_vector,self.danger_threshold)
            
            #print('J_coll is',J_coll)

            print('without scaling',self.qdot_max)
            qdot_max_scaled =multiply(self.qdot_max,self.scaled_maximum_vector)

            print('qdot_max_scaled',qdot_max_scaled)

            polytope_verts_hsm, polytope_faces_hsm = cartesian_velocity_polytope(J_Hess,self.qdot_min,qdot_max_scaled)


            

            ########### Obstacle with HSM POlytope plot ###########################################################################
            # Publish polytope faces
            polyArray_cmp_hsm_message = self.publish_velocity_cmp_hsm.publish(create_polytopes_msg(polytope_verts_hsm, polytope_faces_hsm, \
                                                                                                ef_pose,"telebot_cell_base_link", scaling_factor))
            



            # J_coll = get_constraint_obstacle_jacobian(J_Hess, self.obstacle_link_vector, 1)

            
            # polytope_verts_cmp_obs, polytope_faces_cmp_obs = cartesian_velocity_polytope(J_coll,self.qdot_max,qdot_min)
            # polytope_verts_cmp_obs, polytope_faces_cmp_obs = cartesian_velocity_with_obstacle_polytope(J_Hess_obs, self.obstacle_link_vector, 1)

            

            polytope_point_msg = PointStamped()
            polytope_point_msg.header = Header()
            polytope_point_msg.header.frame_id = 'telebot_cell_base_link'
            polytope_point_msg.point.x = pos_act[0]+ polytope_center[0]/scaling_factor
            polytope_point_msg.point.y = pos_act[1]+polytope_center[1]/scaling_factor
            polytope_point_msg.point.z = pos_act[2]+polytope_center[2]/scaling_factor


            #self.chebychev_msg.publish(polytope_point_msg)

            ef_pose = transpose(pos_act)
            #ef_pose = transpose(polytope_center)
            viz.display(self.q_in_numpy)



            #print('ellipsoid ball is',ellipsoid_ball)
            #input('stop now and see')

            #print('polytope_verts',polytope_verts)
            ########### Actual POlytope plot ###########################################################################
            # Publish polytope faces
            # polyArray_message = self.publish_velocity_polytope.publish(create_polytopes_msg(polytope_verts, polytope_faces, \
            #                                                                                     ef_pose,"telebot_cell_base_link", scaling_factor))
            
            polyArray_cmp_message = self.publish_velocity_cmp.publish(create_polytopes_msg(self.polytope_verts_cmp, self.polytope_faces_cmp, \
                                                                                                ef_pose,"telebot_cell_base_link", scaling_factor))
            

            time_end = rospy.Time.now()
            duration1 = time_end - time_begin
            print('Duration for computation',duration1)


            end_ef_kuka_msg = PointStamped()
            end_ef_kuka_msg.header = Header()

            end_ef_kuka_msg.header.frame_id = 'telebot_cell_base_link'
            end_ef_kuka_msg.header.stamp = rospy.Time.now()

            # frame_info = self.rmodel.frames[self.grasp_frame_kuka_id]
            # print(frame_info.placement.translation)

            # pos_midpoiint = pin.Transform3f()
            # pos_midpoiint[0] = midpoint[0]
            # pos_midpoiint[1] = midpoint[1]
            # pos_midpoiint[2] = midpoint[2]
            #self.rmodel.frames[self.grasp_frame_kuka_id].positionInParentFrame()
            #self.rdata.oMf[self.rmodel.getFrameId('grasp_frame_r1')]#*

            # print('intermmediate frame is',self.grasp_frame_kuka_id)
            # grasp_frame = self.rmodel.frames[self.grasp_frame_kuka_id].placement
            

            


            #print(grasp_frame)
            end_ef_kuka_msg.point.x = pos_act[0]
            end_ef_kuka_msg.point.y = pos_act[1]
            end_ef_kuka_msg.point.z = pos_act[2]
            self.end_effector_position_kuka.publish(end_ef_kuka_msg)




    def rosbag_start_record_callback(self,start_rosbag_record):
        if(start_rosbag_record.data):
            try:
                startRosbagRecordSrv = rospy.ServiceProxy('/data_recording/start_recording', Trigger, persistent=True)
                resp2 = startRosbagRecordSrv()
                

                if resp2.success:
                    print("Rosbag Start Recording Service started ")
                    
                    
                    return resp2.success
                else:
                    print("Rosbag Start Recording Service Error ")
                    return resp2.success

                
            except rospy.ServiceException:
                print("Rosbag Start Recording Service call failed")
    

    def rosbag_stop_record_callback(self,stop_rosbag_record):
        if(stop_rosbag_record.data):
            try:
                stopRosbagRecordSrv = rospy.ServiceProxy('/data_recording/stop_recording', Trigger, persistent=True)
                resp3 = stopRosbagRecordSrv()
                

                if resp3.success:
                    print("Rosbag Stop Recording Service started ")
                    
                    
                    return resp3.success
                else:
                    print("Rosbag Stop Recording Service Error ")
                    return resp3.success

                
            except rospy.ServiceException:
                print("Rosbag Stop Recording Service call failed")

    def button_robot_update_callback(self, geo_robot_buttons):

        self.button_robot_state = [
            geo_robot_buttons.buttons[0], geo_robot_buttons.buttons[1]]
        #print('buttons: ', self.button_robot_state)

    def kuka_digital_output_service(self, out1, out2, out3, out4, out5, out6, out7, out8):
        # print('Deadly inside')
        rospy.wait_for_service(
            '/kuka_hardware_interface/write_8_digital_outputs', timeout=None)
        print('I crossed timeout')
        try:
            write_8_outputs_func = rospy.ServiceProxy(
                '/kuka_hardware_interface/write_8_digital_outputs', write_8_outputs, persistent=True)
            resp1 = write_8_outputs_func(out1, out2, out3, out4, out5, out6, out7, out8)
            print("I have actuated it ")
            return resp1
            # rospy.spin()
            # resp1 = write_8_bool_outputs_resp(False,False,False,False,False,False,False,False)
        except rospy.ServiceException:
            print("Service call failed: KUKA_Digital_outputs")

    def collision_state_callback(self, collision_state):
        self.current_collision_state = collision_state.data
        # print('collision_state is', self.current_collision_state)

    

            # self.changing_state=False
    def gripper_actuate_callback(self, change_gripper_states):

                        # Gripper close here
        gripper_msg = change_gripper_states
        if (change_gripper_states.data):
        # if (self.change_gripper_state and self.changing_state):
            print("\n\n====================Changing state now===============\n\n")
            if(self.gripper_closed):
                print('Open Gripper')

                self.kuka_digital_output_service(
                    True, False, False, False, False, False, False, False)
                rospy.sleep(0.75)
                self.kuka_digital_output_service(
                    False, False, False, False, False, False, False, False)
                gripper_msg.data = False
                self.meca_gripper_state_publisher.publish(change_gripper_states.data)
                self.gripper_closed = False
                print('Finish Open Gripper')
            else:
                print('Close Gripper')

                # self.kuka_digital_output_service(False,False,False,False,False,False,False,False)
                self.kuka_digital_output_service(
                    False, True, False, False, False, False, False, False)
                rospy.sleep(0.75)
                self.kuka_digital_output_service(
                    False, False, False, False, False, False, False, False)
                
                # Close mecademic gripper

                self.meca_gripper_state_publisher.publish(change_gripper_states.data)
                self.gripper_closed = True
                print('Finish Close Gripper')
            print('\n\n====================Changing state back to false===============\n\n')
            


    '''
    def gripper_close_callback(self, gripper_states):
        print("\n\n====================Trying to Close Gripper ===============\n\n")
        if (gripper_states.data):
            print("\n\n====================Closing Gripper ===============\n\n")

            # self.kuka_digital_output_service(False,False,False,False,False,False,False,False)
            self.kuka_digital_output_service(
                True, False, False, False, False, False, False, False)
            rospy.sleep(1.5)
            self.kuka_digital_output_service(
                False, False, False, False, False, False, False, False)
            rospy.sleep(1.5)
            # self.gripper_closed = True
            print('Finish Close Gripper')

            # self.changing_state=False
    '''
    def meca_callback(self, meca_qin_joints):
        # Callback for getting current joint states of MECA
        mutex.acquire()
        #self.q_in[0] = meca_qin_joints.position[0]
        self.q_in_numpy[6] = meca_qin_joints.position[0]
        #self.q_in[1] = meca_qin_joints.position[1]
        self.q_in_numpy[7] = meca_qin_joints.position[1]
        #self.q_in[2] = meca_qin_joints.position[2]
        self.q_in_numpy[8] = meca_qin_joints.position[2]
        #self.q_in[3] = meca_qin_joints.position[3]
        self.q_in_numpy[9] = meca_qin_joints.position[3]
        #self.q_in[4] = meca_qin_joints.position[4]
        self.q_in_numpy[10] = meca_qin_joints.position[4]
        #self.q_in[5] = meca_qin_joints.position[5]
        self.q_in_numpy[11] = meca_qin_joints.position[5]
        mutex.release()

    def kuka_callback(self, kuka_qin_joints):
        # Callback for getting current joint states of KUKA
        mutex.acquire()
        #self.q_in[0] = kuka_qin_joints.position[0]
        self.q_in_numpy[0] = kuka_qin_joints.position[0]
        #self.q_in[1] = kuka_qin_joints.position[1]
        self.q_in_numpy[1] = kuka_qin_joints.position[1]
        #self.q_in[2] = kuka_qin_joints.position[2]
        self.q_in_numpy[2] = kuka_qin_joints.position[2]
        #self.q_in[3] = kuka_qin_joints.position[3]
        self.q_in_numpy[3] = kuka_qin_joints.position[3]
        #self.q_in[4] = kuka_qin_joints.position[4]
        self.q_in_numpy[4] = kuka_qin_joints.position[4]
        #self.q_in[5] = kuka_qin_joints.position[5]
        self.q_in_numpy[5] = kuka_qin_joints.position[5]
        mutex.release()

    def geo_to_robot_callback(self, geo_robot_twist):
        # Callback for Twist control
        # rospy.wait_for_service('write_8_outputs')
        
        #print('no  of jointss is',self.no_of_joints)
        if self.EnterFlag == True:

            self.kuka_joint_states.position = [0.0,0.0,0.0,0.0,0.0,0.0]
            self.meca_joint_states.position = [0.0,0.0,0.0,0.0,0.0,0.0]
            self.EnterFlag = False

            pin.forwardKinematics(self.rmodel,self.rdata, self.q_in_numpy)
            pin.updateFramePlacements(self.rmodel,self.rdata)
            # self.grasp_frame_r1_SE3 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')]
            # print('self.rdata.oMf[self.rmodel.getFrameId_tcp_kuka',self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')])
            # self.grasp_frame_r1_SE3.translation = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation - self.rdata.oMf[self.rmodel.getFrameId('tool0')].translation +  \
            # (self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation - self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation)*0.5
            # print('self.grasp_frame_r1_SE3.translation',self.grasp_frame_r1_SE3.translation)
            # self.grasp_frame_r2_SE3 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')]
            # self.grasp_frame_r2_SE3.translation = (self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation - self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation)*0.5

            # print('self.grasp_frame_r2_SE3.translation',self.grasp_frame_r2_SE3.translation)
            
            # self.grasp_frame_kuka_id = self.rmodel.addFrame(pin.Frame('grasp_frame_r1',6,0,self.grasp_frame_r1_SE3,pin.FrameType.OP_FRAME)) # Returns the ID of the frame
            # self.grasp_frame_meca_id= self.rmodel.addFrame(pin.Frame('grasp_frame_r2',12,0,self.grasp_frame_r2_SE3,pin.FrameType.OP_FRAME)) # Returns the ID of the frame
                    
            # frame_info = self.rmodel.frames[self.grasp_frame_kuka_id]
            # print(frame_info.placement.translation)


            
        for k in range(6):
            self.kuka_joint_states.position[k] = self.q_in_numpy[k]
            # Getting the mecademic joint position here
            self.meca_joint_states.position[k] = self.q_in_numpy[6+k]

        dt = geo_robot_twist.header.stamp.secs

        # Gripper state message

        self.gripper_state_msg.data = self.gripper_closed
        self.kuka_gripper_current_state.publish(self.gripper_state_msg)

        ##### Linear Velocity control here ###############################

        if (dt > 0.000001)  and (self.button_robot_state[0] == 1) and (self.button_robot_state[1] == 0):
            #print('self.force_baseline_arr_meca',self.force_baseline_arr_meca)
            #print('force norm is',self.force_norm)
            #print('torque norm is',self.torque_norm)
            if (self.force_norm) > 7.0 or (self.torque_norm > 7.0):
                rospy.logerr("Forces and torque exceeded.")
                input('Press-space bar to release the Object')
            else:
                print('started control')
            
            #print('Linear vell')
            
            ### Message for Mecademic trajectory
            mutex3.acquire()
            meca_msg = JointState()
            #meca_joints = JointTrajectoryPoint() 
            meca_msg.name = ['meca_axis_1_joint','meca_axis_2_joint','meca_axis_3_joint','meca_axis_4_joint','meca_axis_5_joint','meca_axis_6_joint'] 
            meca_msg.header = geo_robot_twist.header
            meca_msg.header.frame_id='base'

            ### Message for kuka trajectory
            kuka_msg = JointTrajectory()
            kuka_joints = JointTrajectoryPoint()
            kuka_msg.joint_names = self.robot_joint_names[0:6]
            # msg.header = geo_kuka_twist.header

            kuka_msg.header.stamp = rospy.Time.now()
            kuka_msg.header.frame_id = 'base_kuka'

            if self.flag_linear == False:
                self.flag_linear = True
                self.lin_vel_x_prev = 0
                self.lin_vel_y_prev = 0
                self.lin_vel_z_prev = 0

            self.cartesian_twist[0] = self.lin_vel_x_prev + 0.000004 * \
                (geo_robot_twist.twist.linear.x - self.lin_vel_x_prev)
            self.cartesian_twist[1] = self.lin_vel_y_prev + 0.000004 * \
                (geo_robot_twist.twist.linear.y - self.lin_vel_y_prev)
            self.cartesian_twist[2] = self.lin_vel_z_prev + 0.000004 * \
                (geo_robot_twist.twist.linear.z - self.lin_vel_z_prev)

            self.lin_vel_x_prev = self.cartesian_twist[0]
            self.lin_vel_y_prev = self.cartesian_twist[1]
            self.lin_vel_z_prev = self.cartesian_twist[2]

            self.cartesian_twist[0] = self.cartesian_twist[0]*self.vel_scale[0]
            self.cartesian_twist[1] = self.cartesian_twist[1]*self.vel_scale[1]
            self.cartesian_twist[2] = self.cartesian_twist[2]*self.vel_scale[2]
            


            self.cartesian_twist[3] = 0.0
            self.cartesian_twist[4] = 0.0
            self.cartesian_twist[5] = 0.0
        
            # Using pinnochio 

            '''
            pin.updateGeometryPlacements(self.rmodel,self.rdata,self.geom_model,self.geom_data,self.q_in_numpy)
            pin.computeDistances(self.rmodel, self.rdata, self.geom_model, self.geom_data, self.q_in_numpy)
            distance_results = self.geom_data.distanceResults
            #print('distanceresults',distance_results)
            # Print the status of collision for all collision pairs
            counter = 0

            # msg1 = PointStamped()
            # msg1.header = Header()
            # msg1.header.frame_id = 'telebot_cell_base_link'
            # msg2 = PointStamped()
            # msg2.header = Header()
            # msg2.header.frame_id = 'telebot_cell_base_link'

            for result in distance_results: 
                #cr = geom_data.collisionResults[k].closestPoints
                #dr_result = distance_results[k]
                #print('result is',dir(result))
                #print('o1',result.min_distance)

                
                
                #cp = robot.collision_model.collisionPairs[k]
                cp1 = result.getNearestPoint1()
                cp2 = result.getNearestPoint2()
                #print('norm of two points',norm(cp2-cp1))
                #input('stop now')
                #print('closest points 1',result.getNearestPoint1())
                #print('closest points 2',result.getNearestPoint2())


                # msg1.point.x = cp1[0]
                # msg1.point.y = cp1[1]
                # msg1.point.z = cp1[2]

                # msg2.point.x = cp2[0]
                # msg2.point.y = cp2[1]
                # msg2.point.z = cp2[2]



                if (counter < 12):
                    self.obstacle_dist_vector[counter] = result.min_distance
                    if result.min_distance <= self.danger_threshold:

                        self.obstacle_link_vector[counter,0] = cp2[0] - cp1[0]
                        self.obstacle_link_vector[counter,1] = cp2[1] - cp1[1]
                        self.obstacle_link_vector[counter,2] = cp2[2] - cp1[2]
                        print('self.obstacle link')
                        # self.cp1_publisher_msg[counter].publish(msg1)
                        # self.cp2_publisher_msg[counter].publish(msg2)

                counter += 1

            
            pin.forwardKinematics(self.rmodel,self.rdata,self.q_in_numpy)
            
            pin.updateFramePlacements(self.rmodel,self.rdata)
            pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

            
            J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_kuka'), pin.LOCAL_WORLD_ALIGNED)
            J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
            J_Hess = hstack((J_Hess1[:,:6],J_Hess2[:,6:]))
            
                    
            pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
            pos_act2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation

            print('pos of kuka is',pos_act1)
            print('pos of meca is',pos_act2)

            pos_act = pos_act1 + (pos_act2 - pos_act1)*0.5
            ef_pose = transpose(pos_act)
            

            scaling_factor = 80.0
            '''
            ### Polytope plot with estimation
            
            '''
            polytope_verts, polytope_faces = cartesian_velocity_polytope(J_Hess,self.qdot_max,self.qdot_min)
            


            

            
            ########### Actual POlytope plot ###########################################################################
            # Publish polytope faces
            polyArray_message = self.publish_velocity_polytope.publish(create_polytopes_msg(polytope_verts, polytope_faces, \
                                                                                                ef_pose,"telebot_cell_base_link", scaling_factor))
            
            
            polytope_verts_cmp, polytope_faces_cmp,polytope_center,polytope_center_max = cartesian_cmp_polytope(J_Hess,self.q_in_numpy,self.qdot_min,self.qdot_max,\
                                                                                            self.q_min,self.q_max,self.q_mean,self.psi_max, self.psi_min, \
                                                                                                self.obstacle_link_vector, \
                                                                                                    self.obstacle_dist_vector,self.danger_threshold)
            

            
            # polytope_verts_cmp, polytope_faces_cmp = cartesian_velocity_with_joint_limit_polytope(J_Hess,self.q_in_numpy,self.qdot_min,self.qdot_max,\
            #                                                                         self.q_min,self.q_max,self.q_mean)
        

            if polytope_verts_cmp.any():
                self.polytope_verts_cmp = polytope_verts_cmp
                self.polytope_faces_cmp = polytope_faces_cmp
            polyArray_cmp_message = self.publish_velocity_cmp.publish(create_polytopes_msg(self.polytope_verts_cmp, self.polytope_faces_cmp, \
                                                                                                ef_pose,"telebot_cell_base_link", scaling_factor))
            
            
            
            # polytope_point_msg = PointStamped()
            # polytope_point_msg.header = Header()
            # polytope_point_msg.header.frame_id = 'telebot_cell_base_link'
            # polytope_point_msg.point.x = pos_act[0]+ polytope_center[0]/scaling_factor
            # polytope_point_msg.point.y = pos_act[1]+polytope_center[1]/scaling_factor
            # polytope_point_msg.point.z = pos_act[2]+polytope_center[2]/scaling_factor



            #ef_pose = transpose(polytope_center)
            viz.display(self.q_in_numpy)
            '''


            #print('ellipsoid ball is',ellipsoid_ball)
            #input('stop now and see')

            #print('polytope_verts',polytope_verts)
            ########### Actual POlytope plot ###########################################################################
            # Publish polytope faces
            # polyArray_message = self.publish_velocity_polytope.publish(create_polytopes_msg(polytope_verts, polytope_faces, \
            #                                                                                     ef_pose,"telebot_cell_base_link", scaling_factor))
            
            
            # polyArray_cmp_obs_message = self.publish_velocity_cmp_obs.publish(create_polytopes_msg(polytope_lower_cmp, poly_faces_lower_cmp, \
            #                                                                                     ef_pose,"telebot_cell_base_link", scaling_factor))
            
            # polyArray_cmp_obs_message = self.publish_velocity_cmp_obs.publish(create_polytopes_msg(polytope_verts_cmp_obs, polytope_faces_cmp_obs, \
            #                                                                                     ef_pose,"telebot_cell_base_link", scaling_factor))
            

            
            
            pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

            

            
            J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_kuka'), pin.LOCAL_WORLD_ALIGNED)
            J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)

            print('J_Hess1',J_Hess1)
            pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
            pos_act2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation

            print('pos of kuka is',pos_act1)
            print('pos of meca is',pos_act2)
            
            # J_Hess = hstack((J_Hess1[:,:6],J_Hess2[:,6:]))

            #aaa = self.rmodel.addFrame(pin.Frame('grasp_frame_r1',6,self.rmodel.getFrameId('tcp_kuka'),self.grasp_frame_r1_SE3,pin.FrameType.OP_FRAME))

            #print('J_Hess1',J_Hess1)
            #print('sself.carteesian_twist',self.cartesian_twist)
            #print('J_Hess1',pinv(J_Hess1).dot(self.cartesian_twist))

            self.qdot_out[0:6] = pinv(J_Hess1).dot(self.cartesian_twist)[0:6]
            #print('self.qdot_out using pinochio is',self.qdot_out)
            #print('J_Hess2',pinv(J_Hess2).dot(self.cartesian_twist))
            self.qdot_out[6:] = pinv(J_Hess2).dot(self.cartesian_twist)[6:]
            #print('self.qdot_out using pinochio is',self.qdot_out)


            print('error is',norm(pos_act2 - pos_act1))
            #v = -J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), desired_ee_vel)
            ## I have commented below

            
            for no_of_joints in range(self.no_of_joints):
                if (abs(self.qdot_out[no_of_joints]) - abs(self.qdot_limit[no_of_joints])) > 0.050000:
                    for i in range(self.rmodel.njoints-1):
                        self.qdot_out[i] = 0.0                        
                        print('Torque error')
                        input('wait here - Torque error')
                    return
            for no_of_joints in range(self.no_of_joints-6):
                if ((self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints]) >
                self.q_upper_limit[no_of_joints]) or ((self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints])
                 < self.q_lower_limit[no_of_joints]):
                    self.qdot_out[no_of_joints] = 0
                if ((self.meca_joint_states.position[no_of_joints] + self.qdot_out[6+no_of_joints]) >
                self.q_upper_limit[6+no_of_joints]) or ((self.meca_joint_states.position[no_of_joints] + self.qdot_out[6+no_of_joints])
                 < self.q_lower_limit[6+no_of_joints]):
                    self.qdot_out[6+no_of_joints] = 0



            for no_of_joints in range(self.no_of_joints-6):
                self.kuka_joint_states.position[no_of_joints] = self.kuka_joint_states.position[no_of_joints] + \
                    self.qdot_out[no_of_joints]
                #print('my kuka joints are',self.kuka_joint_states.position)
                
                self.meca_joint_states.position[no_of_joints] = self.meca_joint_states.position[no_of_joints] + \
                    self.qdot_out[6+no_of_joints]


            
            
            self.kuka_joint_states.header = geo_robot_twist.header
            kuka_joints.positions = self.kuka_joint_states.position

            kuka_joints.accelerations = []
            kuka_joints.effort = []
            kuka_joints.time_from_start = rospy.Duration(0, 100000000)
            
            kuka_msg.points.append(kuka_joints)

            

            # self.kuka_joint_states_dummy_publisher.publish(msg)
            
            self.kuka_joint_states_publisher.publish(kuka_msg)



            meca_msg.position = self.meca_joint_states.position
            self.meca_joint_states_publisher.publish(meca_msg)




            mutex3.release()


            mutex3.acquire()
            # q_in_interm = zeros(12)
            # new_cartesian_twist = zeros(6)
            # for k in range(6):
            #     q_in_interm[k] = self.kuka_joint_states.position[k]
            #     q_in_interm[6+k] = self.meca_joint_states.position[k]
            pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)
            pin.framesForwardKinematics(self.rmodel, self.rdata, self.q_in_numpy)
            pos_act_int1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
            pos_act_int2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation

            print('pos_act1',pos_act_int1)
            print('pos_act2',pos_act_int2)
            pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)
            #print('self.q_in_numpy is',self.q_in_numpy)
            J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
                                              
            oMdes = pin.SE3(eye(3), array([pos_act_int1[0], pos_act_int1[1], pos_act_int1[2]]))
            OMcurr = pin.SE3(eye(3), array([pos_act_int2[0], pos_act_int2[1], pos_act_int2[2]]))

            dMi = oMdes.actInv(OMcurr)
            err = pin.log(dMi).vector

            if norm(err) < 3e-3:
                success = True
            else:
                

                J = J_Hess2
                #print('Jacobian is',J)
                #print('err',err)
                #v = -J_Hess2.T.dot(solve(J_Hess2.dot(J_Hess2.T) + self.damp * eye(6), err))

                #q_out_meca = pin.integrate(self.rmodel,self.q_in_numpy,v*self.DT)

                #new_cartesian_twist[0:3] = pos_act_int1 - pos_act_int2
                #print('solve(J.dot(transpose(J)) + (self.damp*eye(6))',J.dot(transpose(J)) + eye(6))

                #self.qdot_out_meca[6:] = -transpose(J).dot(solve(J.dot(transpose(J)) + (self.damp*eye(6)), err))[6:]
                self.qdot_out_meca[6:] = -pinv(J_Hess2).dot(err)[6:]
                print('self.qdot_out_meca is',self.qdot_out_meca)
                print('self.qdot_limit',self.qdot_limit)
                print(self.no_of_joints)

                for no_of_joints in range(6,self.no_of_joints):
                    print('self.qdot_out_meca[no_of_joints]',self.qdot_out_meca[no_of_joints])
                    if (abs(self.qdot_out_meca[no_of_joints]) - abs(self.qdot_limit[no_of_joints])) > 0.050000:
                        for i in range(6,self.rmodel.njoints-1):
                            self.qdot_out[i] = 0.0                        
                            print('Torque error')
                            input('wait here - Torque error')
                        return
                for no_of_joints in range(self.no_of_joints-6):

                    if ((self.meca_joint_states.position[no_of_joints] + self.qdot_out_meca[6+no_of_joints]) >
                    self.q_upper_limit[6+no_of_joints]) or ((self.meca_joint_states.position[no_of_joints] + self.qdot_out_meca[6+no_of_joints])
                    < self.q_lower_limit[6+no_of_joints]):
                        self.qdot_out_meca[6+no_of_joints] = 0



                for no_of_joints in range(self.no_of_joints-6):
                    #print('my kuka joints are',self.kuka_joint_states.position)
                    
                    self.meca_joint_states.position[no_of_joints] = self.meca_joint_states.position[no_of_joints] + \
                        self.qdot_out_meca[6+no_of_joints]
                


                meca_msg.position = self.meca_joint_states.position
                self.meca_joint_states_publisher.publish(meca_msg)

                
            
             


            
            mutex3.release()
            #pin.framesForwardKinematics(self.rmodel, self.rdata, self.q_in_numpy)
            #pin.updateFramePlacements(self.rmodel,self.rdata)

            #pin.framesForwardKinematics(self.rmodel,self.rdata)
            
            #print('grasp frame id',self.rmodel.getFrameId('grasp_frame_r1'))
            #pin.updateFramePlacement(self.rmodel,self.rdata,self.rmodel.getFrameId('grasp_frame_r1'))
            #print(self.rdata.oMi[self.rmodel.getFrameId('tcp_kuka')]*self.rdata.iMf[self.rmodel.getFrameId('grasp_frame_r1')])
            #print(self.rdata.oMf[37])

            #pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('grasp_frame_r1')].translation
            # pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
            # pos_act2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation

            # print('ppos_act1',pos_act1)
            # print('ppos_act2',pos_act2)

            # midpoint = (pos_act2-pos_act1)/2.0 + pos_act1
            # print('midpoint',(pos_act2-pos_act1)/2.0 + pos_act1)

            #kuka_frame = self.rmodel.frames[self.grasp_frame_kuka_id].placement.translation
            #meca_frame = self.rmodel.frames[self.grasp_frame_meca_id].placement.translation
            #print('KUKA end-effector-position is',self.rmodel.frames[self.grasp_frame_kuka_id].placement.translation)
            #print('kuka position in world',self.rdata.oMi[6]*self.rmodel.frames[self.grasp_frame_kuka_id].placement)
            #print('MECA end-effector-position is',self.rmodel.frames[self.grasp_frame_meca_id].placement.translation)
            #print('meca position in world',self.rdata.oMi[12]*self.rmodel.frames[self.grasp_frame_meca_id].placement)
            
            
            




            #print('error between two frames is',norm(pos_act1 - pos_act2))


            # end_ef_kuka_msg = PointStamped()
            # end_ef_kuka_msg.header = Header()

            # end_ef_kuka_msg.header.frame_id = 'telebot_cell_base_link'
            # end_ef_kuka_msg.header.stamp = rospy.Time.now()

            # # frame_info = self.rmodel.frames[self.grasp_frame_kuka_id]
            # # print(frame_info.placement.translation)

            # # pos_midpoiint = pin.Transform3f()
            # # pos_midpoiint[0] = midpoint[0]
            # # pos_midpoiint[1] = midpoint[1]
            # # pos_midpoiint[2] = midpoint[2]
            # #self.rmodel.frames[self.grasp_frame_kuka_id].positionInParentFrame()
            # #self.rdata.oMf[self.rmodel.getFrameId('grasp_frame_r1')]#*

            # # print('intermmediate frame is',self.grasp_frame_kuka_id)
            # # grasp_frame = self.rmodel.frames[self.grasp_frame_kuka_id].placement
            

            


            # #print(grasp_frame)
            # end_ef_kuka_msg.point.x = pos_act1[0]
            # end_ef_kuka_msg.point.y = pos_act1[1]
            # end_ef_kuka_msg.point.z = pos_act1[2]
            # self.end_effector_position_kuka.publish(end_ef_kuka_msg)

            # counter = 0
            # for f in self.rmodel.joints:
            #     print('f',f)
            #     print('f-naame',f.name)
            #     print('counter',counter)
            #     counter+=1

            #3input('stop now')

            #self.end_effector_position_kuka.publish()  


            


            # end_ef_meca_msg = PointStamped()
            # end_ef_meca_msg.header = Header()

            # end_ef_meca_msg.header.frame_id = 'base_link'
            # end_ef_meca_msg.header.stamp = rospy.Time.now()
            # end_ef_meca_msg.point.x = pos_act2[0]
            # end_ef_meca_msg.point.y = pos_act2[1]
            # end_ef_meca_msg.point.z =pos_act2[2]

            # self.end_effector_position_meca.publish(end_ef_meca_msg)

        ##### Angular Velocity control here ###############################
        
        # elif (dt > 0.000001) and self.button_robot_state[1] == 1 and self.button_robot_state[0] == 0:

        #     msg = JointTrajectory()
        #     kuka_joints = JointTrajectoryPoint()
        #     msg.joint_names = ['joint_a1', 'joint_a2',
        #         'joint_a3', 'joint_a4', 'joint_a5', 'joint_a6']
        #     # msg.header = geo_kuka_twist.header
        

        #     msg.header.stamp = rospy.Time.now()

        #     msg.header.frame_id = 'tool0_kuka'

        #     if self.flag_angular == False:
        #         self.flag_angular = True
        #         self.ang_vel_x_prev = 0
        #         self.ang_vel_y_prev = 0
        #         self.ang_vel_z_prev = 0
        #     self.cartesian_twist.rot[0] = self.ang_vel_x_prev + 0.000004 * \
        #         (geo_robot_twist.twist.angular.x - self.ang_vel_x_prev)
        #     self.cartesian_twist.rot[1] = self.ang_vel_y_prev + 0.000004 * \
        #         (geo_robot_twist.twist.angular.y - self.ang_vel_y_prev)
        #     self.cartesian_twist.rot[2] = self.ang_vel_z_prev + 0.000004 * \
        #         (geo_robot_twist.twist.angular.z - self.ang_vel_z_prev)

        #     self.ang_vel_x_prev = self.cartesian_twist.rot[0]
        #     self.ang_vel_y_prev = self.cartesian_twist.rot[1]
        #     self.ang_vel_z_prev = self.cartesian_twist.rot[2]

        #     self.cartesian_twist.rot[0] = self.cartesian_twist.rot[0] * \
        #         self.angular_vel_scale[0]
        #     self.cartesian_twist.rot[1] = self.cartesian_twist.rot[1] * \
        #         self.angular_vel_scale[1]
        #     self.cartesian_twist.rot[2] = self.cartesian_twist.rot[2] * \
        #         self.angular_vel_scale[2]
            
        #     self.cartesian_twist.vel[0] = 0.0
        #     self.cartesian_twist.vel[1] = 0.0
        #     self.cartesian_twist.vel[2] = 0.0
        #     self.vel_fk_solver.JntToCart(self.q_in, self.eeFrame)
        #     R_B_tool = self.eeFrame.M ### R^B_t
        #     #print('R_B_tool',R_B_tool)
        #     cartesian_twist_transform = R_B_tool*self.cartesian_twist

        #     #self.vel_ik_solver.CartToJnt(self.q_in, cartesian_twist_transform, self.qdot_out)

        #     self.vel_ik_solver.CartToJnt(self.q_in, self.cartesian_twist, self.qdot_out)

        #     print('self.qdot_out using KDL is',self.qdot_out)

            
        #     # Using pinnochio 

            
        #     pin.forwardKinematics(self.rmodel,self.rdata,self.q_in_numpy)
            
        #     #nu = pin.log(Mtool.inverse() * Mgoal).vector
            
        #     ## Tool velocity is here

        #     eps    = 1e-4
        #     IT_MAX = 1000
        #     DT     = 1e-2
        #     damp   = 1e-12
        #     nu = R_B_tool*self.cartesian_twist

        #     nu = pin.log(pin.SE3.Identity()).vector

        #     nu[0] = self.cartesian_twist.vel[0] 
        #     nu[1] = self.cartesian_twist.vel[1] 
        #     nu[2] = self.cartesian_twist.vel[2] 

        #     nu[3] = self.cartesian_twist.rot[0] 
        #     nu[4] = self.cartesian_twist.rot[1] 
        #     nu[5] = self.cartesian_twist.rot[2] 


        #     pin.computeFrameJacobian(self.rmodel, self.rdata,self.q_in_numpy,self.rmodel.getFrameId('tool_meca'))
        #     pin.forwardKinematics(self.rmodel,self.rdata, self.q_in_numpy)
        #     pin.updateFramePlacements(self.rmodel,self.rdata)

        #     #print('self.rmodel.getJointId',self.rmodel.getJointId('tool0'))
        #     #print('self.rmodel.grasp_frame',self.rmodel.getFrameId(33))
        #     pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
        #     pos_act2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation

        #     pos_act = (pos_act1 + pos_act2)*0.5


            



            
        #     pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

            
        #     J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('grasp_frame_r1'), pin.LOCAL_WORLD_ALIGNED)
        #     J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('grasp_frame_r1'), pin.LOCAL_WORLD_ALIGNED)
            
        #     J_Hess = hstack((J_Hess1[:,:6],J_Hess2[:,6:]))            

        #     #print('J_Hess1',J_Hess1)
            
        #     qdot_out = pinv(J_Hess2).dot(nu)
        #     print('self.qdot_out using pinochio is',qdot_out)

        #     print('errror is', norm(self.rdata.oMf[self.rmodel.getFrameId('grasp_frame_r1')].translation-\
        #           self.rdata.oMf[self.rmodel.getFrameId('grasp_frame_r2')].translation))

            

        #     for no_of_joints in range(6):
        #         if (abs(self.qdot_out[no_of_joints]) - self.qdot_limit[no_of_joints]) > 0.050000:
        #             for i in range(self.rmodel.njoints):
        #                 self.qdot_out[i] = 0.0
        #                 print('Torque error')
        #             return
        #         if ((self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints]) >
        #         self.q_upper_limit[no_of_joints]) or ((self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints])
        #          < self.q_lower_limit[no_of_joints]):
        #             self.qdot_out[no_of_joints] = 0

        #         # self.kuka_joint_states.position[no_of_joints] = self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints]

        #         # self.q_in[no_of_joints] = self.kuka_joint_states.position[no_of_joints]

        #     for no_of_joints in range(6):
        #         self.kuka_joint_states.position[no_of_joints] = self.kuka_joint_states.position[no_of_joints] + \
        #             self.qdot_out[no_of_joints]
        #         # self.q_in[no_of_joints] = self.kuka_joint_states.position[no_of_joints]

        #     self.kuka_joint_states.header = geo_robot_twist.header
        #     kuka_joints.positions = self.kuka_joint_states.position

        #     kuka_joints.accelerations = []
        #     kuka_joints.effort = []
        #     kuka_joints.time_from_start = rospy.Duration(0, 400000000)

        #     # kuka_joints.time_from_start.nsecs = 778523489 # Taken from rqt_joint_trajectory_controller
        #     msg.points.append(kuka_joints)

            
        
            


        else:

            self.flag_linear = False
            self.flag_angular = False

            
            msg = JointTrajectory()
            kuka_joints = JointTrajectoryPoint()
            msg.joint_names = self.robot_joint_names
            # msg.header = geo_kuka_twist.header

            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'base_kuka'

            

            self.kuka_joint_states.header = geo_robot_twist.header
            kuka_joints.positions = self.kuka_joint_states.position

            kuka_joints.accelerations = []
            kuka_joints.effort = []
            kuka_joints.time_from_start = rospy.Duration(0, 400000000)

            # kuka_joints.time_from_start.nsecs = 778523489 # Taken from rqt_joint_trajectory_controller
            msg.points.append(kuka_joints)




        
        rate = rospy.Rate(self.pub_rate) # Hz
        rate.sleep()
    def fmin_opt_ik(self,x0_start,pose_desired,analytical_solver: bool):
        self.initial_x0 = float64(x0_start)
        self.pos_reference = float64(pose_desired)
        print('Reference position is', self.pos_reference)

        # input('self.pos_reference')

        # Desired vertex set

        # + self.pos_reference[0,0]
        self.desired_vertices[:, 0] = self.cartesian_desired_vertices[:, 0]
        # + self.pos_reference[0,1]
        self.desired_vertices[:, 1] = self.cartesian_desired_vertices[:, 1]
        # + self.pos_reference[0,2]
        self.desired_vertices[:, 2] = self.cartesian_desired_vertices[:, 2]
        # print('self.opt_polytope_gradient_model.d_gamma_hat',self.opt_polytope_gradient_model.d_gamma_hat)

        # Bounds created from the robot angles

        self.opt_bounds = float64(self.q_bounds)
        print('q_bounds are',self.opt_bounds)
        #q      = pin.neutral(self.rmodel)
        eps    = 1e-5
        IT_MAX = 1000
        DT     = 1e-1
        damp   = 1e-12

        i=0
        self.fun_iter.data = 100

        while True:
            pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)
            pin.framesForwardKinematics(self.rmodel, self.rdata, self.q_in_numpy)
            #pos_act_int1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
            pos_act_int1 = pose_desired
            pos_act_int2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation

            print('pos_act1',pos_act_int1)
            print('pos_act2',pos_act_int2)
            pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)
            #print('self.q_in_numpy is',self.q_in_numpy)
            J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
            J = J_Hess2                          
            oMdes = pin.SE3(eye(3), array([pos_act_int1[0], pos_act_int1[1], pos_act_int1[2]]))
            OMcurr = pin.SE3(eye(3), array([pos_act_int2[0], pos_act_int2[1], pos_act_int2[2]]))

            dMi = oMdes.actInv(OMcurr)
            err = pin.log(dMi).vector
            self.fun_counter += 0.5
            self.fun_iter.data = int(self.fun_counter)

            if norm(err) < eps:
                success = True
                self.msg_status_ik.data = 'Success'
                self.fun_iter.data = int(100)
                self.pub_end_ik.publish(self.fun_iter)
                break
            if i >= IT_MAX:
                success = False
                #self.fun_counter += 0.5
                self.fun_iter.data = int(100)
                self.msg_status_ik.data = 'Time limit'
                break

            v = - J.T.dot(solve(J.dot(J.T) + damp * eye(6), err))
            q = pin.integrate(self.rmodel,self.q_in_numpy,v*DT)

            self.q_in_numpy = q
            self.joint_state_publisher_robot(q)
            viz.display(q0)



            i += 1

            self.pub_status_ik.publish(self.msg_status_ik)
            self.pub_end_ik.publish(self.fun_iter)

        self.pub_status_ik.publish(self.msg_status_ik)
        self.pub_end_ik.publish(self.fun_iter)

        

        
        self.polytope_display = False

        q_joints_opt = q


        return q_joints_opt

    def fmin_opt(self, x0_start, pose_desired, analytical_solver: bool):
        ### Function - func
        # Initial point - x0
        # args -
        ## method - SLQSQ
        # jac = Jacobian - gradient of the

        #self.opt_polytope_model = polytope_model
        #self.opt_polytope_gradient_model = polytope_gradient_model

        #self.q_joints_input = robot_model.q_joints

        #self.obj_function = polytope_model.Gamma_total
        # print('self.obj_function',self.obj_function)

        self.initial_x0 = float64(x0_start)

        #self.initial_x0 = randn(6)
        #print('self.initial_x0 ',self.initial_x0 )
        # print('self.obj_function(robot_model.q_joints)',self.obj_function(robot_model.q_joints))
        #self.func_deriv = polytope_gradient_model.d_gamma_hat

        # self.opt_polytope_gradient_model.compute_polytope_gradient_parameters(self.opt_robot_model,self.opt_polytope_model)
        # self.opt_polytope_gradient_model.Gamma_hat_gradient(sigmoid_slope=self.sigmoid_slope)
        #methods = trust-constr

        #x0 = self.initial_x0
        #x0_d = self.initial_x0 + 1e-5
        #numerical_err = (self.obj_function(x0_d) - self.obj_function(x0))
        #grad_err = (self.jac_func(x0_d) - self.jac_func(x0))

        #print('numerical is:',numerical_err)
        #print('analytical error is',grad_err)
        #assert sco.check_grad(func = self.obj_function, grad = self.jac_func, x0 = self.initial_x0,espilon = 1e-5, direction = 'all',seed = None)
        # Get the end-effector posision
        #self.pos_reference = self.opt_robot_model.end_effector_position

        self.pos_reference = float64(pose_desired)
        print('Reference position is', self.pos_reference)

        # input('self.pos_reference')

        # Desired vertex set

        # + self.pos_reference[0,0]
        self.desired_vertices[:, 0] = self.cartesian_desired_vertices[:, 0]
        # + self.pos_reference[0,1]
        self.desired_vertices[:, 1] = self.cartesian_desired_vertices[:, 1]
        # + self.pos_reference[0,2]
        self.desired_vertices[:, 2] = self.cartesian_desired_vertices[:, 2]
        # print('self.opt_polytope_gradient_model.d_gamma_hat',self.opt_polytope_gradient_model.d_gamma_hat)

        # Bounds created from the robot angles

        self.opt_bounds = float64(self.q_bounds)
        print('q_bounds are',self.opt_bounds)
        #self.opt_bounds = self.q_bounds

        #print('self.opt_bounds is',self.opt_bounds)

        # Constraints

        '''
        cons = ({'type': 'ineq', 'fun': self.constraint_func},\
                {'type': 'ineq', 'fun': self.constraint_func_Gamma})
        '''
        #cons = ({'type': 'ineq', 'fun': self.constraint_function,'jac': self.jac_func})

        cons = ({'type': 'eq', 'fun': self.constraint_function, 'tol': 1e-4} )
               
                 

        #cons = ({'type': 'eq', 'fun': self.constraint_function,'tol':1e-5,},
        #        {'type': 'ineq', 'fun': self.constraint_function_Gamma, 'jac': self.jac_func})

        '''


        cons = ({'type': 'eq', 'fun': self.constraint_function,'tol':1e-6})
          
        print('bounds are',self.opt_bounds)
        input('boundsa rea')
        '''

        '''
        cons = ({'type': 'ineq', 'fun': lambda x:  self.q_joints_input[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
        '''
        # jACOBIAN MATRIX OF THE OBJECTIVE FUNCTION

        # jac = self.opt_polytope_gradient_model.d_gamma_hat,

        # WIth analytical gradient

        '''
        
                    self.q_joints_opt = sco.minimize(fun = self.obj_function,  x0 = self.initial_x0,bounds = self.opt_bounds,\
                                         jac =self.jac_func,method='SLSQP', \
                                             options={'disp': True,'maxiter':500})
        
        '''
        self.fun_iter.data = 0

        if analytical_solver:
            q_joints_opt = sco.minimize(fun=self.obj_function_gamma,  x0=self.initial_x0, bounds=self.opt_bounds[6:,:],
                                        jac=self.jac_func, constraints=cons, method='SLSQP',
                                        options={'disp': True, 'maxiter': 100})  # Paper maximum iterations is 3000
        else:

            q_joints_opt = sco.minimize(fun=self.obj_function_IK,  x0=self.initial_x0, bounds=self.opt_bounds,
                                        constraints=cons, tol=1e-6, method='COBYLA',
                                        options={'disp': True})

        self.fun_iter.data = 100

        self.pub_end_ik.publish(self.fun_iter)
        if q_joints_opt.success:

            self.msg_status_ik.data = 'Success'
        
        elif q_joints_opt.status == int(8):
            self.msg_status_ik.data = 'Directional search error'
        else:
            self.msg_status_ik.data = 'Time limit'            
            
        self.pub_status_ik.publish(self.msg_status_ik)
        print('q_joints_opt', q_joints_opt.x)
        self.polytope_display = False


        return q_joints_opt

    

    def obj_function_gamma(self, q_in):

        from numpy.linalg import det
        from numpy import sum

        self.q_in_numpy[6:] = q_in
        # To publish the joint states to Robot
        #self.joint_state_publisher_robot(q_in)


        pin.computeFrameJacobian(self.rmodel, self.rdata,self.q_in_numpy,33)
        pin.forwardKinematics(self.rmodel,self.rdata, self.q_in_numpy)
        pin.updateFramePlacements(self.rmodel,self.rdata)


        #pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
        pos_act2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation



        #pos_act = pos_act1 + (pos_act2 - pos_act1)*0.5
        pos_act = pos_act2

        
        pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

        
        #J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_kuka'), pin.LOCAL_WORLD_ALIGNED)
        J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
        
        #J_Hess = hstack((J_Hess1[:,:6],J_Hess2[:,6:]))
        J_Hess = J_Hess2[:,6:]


        h_plus, h_plus_hat, h_minus, h_minus_hat, p_plus, p_minus, p_plus_hat, p_minus_hat, n_k, Nmatrix, Nnot = get_polytope_hyperplane(
            J_Hess, active_joints=self.active_joints, cartesian_dof_input=array([True, True, True, False, False, False]), qdot_min=self.qdot_min[6:],
            qdot_max=self.qdot_max[6:], cartesian_desired_vertices=self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input)

        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(
            J_Hess, n_k, h_plus, h_plus_hat, h_minus, h_minus_hat,
            active_joints=self.active_joints, cartesian_dof_input=array([True, True, True, False, False, False]), qdot_min=self.qdot_min[6:],
            qdot_max=self.qdot_max[6:], cartesian_desired_vertices=self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input)

        self.Gamma_min_softmax = Gamma_min_softmax

        print('Gamma now is',self.Gamma_min_softmax)
        #err_pos = abs((norm(pos_act-self.pos_reference) - 1e-6))
        # return err_pos
        #self.plot_capacity_margin_est(Gamma_min_softmax)


        return -1.0*self.Gamma_min_softmax

    def obj_function_IK(self, q_in):

        #from scipy.stats import norm as norm_scip

        #self.joint_state_publisher_robot(q_in)
        #pos_act = array(self.pykdl_util_kin.forward(q_in)[0:3, 3])

        self.q_in_numpy = q_in
        pin.computeFrameJacobian(self.rmodel, self.rdata,self.q_in_numpy,33)
        pin.forwardKinematics(self.rmodel,self.rdata, self.q_in_numpy)
        pin.updateFramePlacements(self.rmodel,self.rdata)


        pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
        pos_act2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation



        pos_act = pos_act1 + (pos_act2 - pos_act1)*0.5

        
        pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

        
        J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_kuka'), pin.LOCAL_WORLD_ALIGNED)
        J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
        
        J_Hess = hstack((J_Hess1[:,:6],J_Hess2[:,6:]))


        '''
        J_Hess = array(self.pykdl_util_kin.jacobian(q_in))
        scaling_factor = 5.0

        mutex.acquire()
        
        ### Polytope plot with estimation
        
        #pykdl_kin_jac = pykdl_util_kin.jacobian(self.q_in_numpy)
        polytope_verts, polytope_faces, facet_vertex_idx, capacity_faces, capacity_margin_proj_vertex, \
            polytope_verts_est, polytope_faces_est, capacity_faces_est, capacity_margin_proj_vertex_est = \
                                        velocity_polytope_with_estimation(J_Hess,self.qdot_max,self.qdot_min,self.desired_vertices,self.sigmoid_slope_input)
        desired_polytope_verts, desired_polytope_faces = desired_polytope(self.desired_vertices)



        #print('facet_vertex_idx',facet_vertex_idx)
        #print('capacity_margin_proj_vertex',capacity_margin_proj_vertex)
        #print('capacity_margin_proj_vertex_est',capacity_margin_proj_vertex_est)
        

        # Only for visualization - Polytope at end-effector - No physical significance
        #ef_pose = pykdl_util_kin.forward(self.q_in_numpy)[:3,3]

        
        

        # Get end-effector of the robot here for the polytope offset

        #ef_pose = position_70(q_in)

        ef_pose = pos_act
        ef_pose = ef_pose[:,0]
        #print('ef_pose is',ef_pose)
        #input('stop to test ef_pose')

        ########### Actual POlytope plot ###########################################################################
        # Publish polytope faces
        polyArray_message = self.publish_velocity_polytope.publish(create_polytopes_msg(polytope_verts, polytope_faces, \
                                                                                            ef_pose,"base_link", scaling_factor))
        
        
        ### Desired polytope set - Publish

        DesiredpolyArray_message = self.publish_desired_polytope.publish(create_polytopes_msg(desired_polytope_verts, desired_polytope_faces, \
                                                                                            ef_pose,"base_link", scaling_factor))


        ### Vertex for capacity margin on the Desired Polytope
        #print('facet_vertex_idx',facet_vertex_idx)
        closest_vertex = self.cartesian_desired_vertices[facet_vertex_idx[0,1]]
        #print('closest_vertex',closest_vertex)

        CapacityvertexArray_message = self.publish_vertex_capacity.publish(create_capacity_vertex_msg(closest_vertex, \
                                                                                    ef_pose, "base_link", scaling_factor))
        
        ### Vertex for capacity margin on the Available Polytope
        CapacityprojvertexArray_message = self.publish_vertex_proj_capacity.publish(create_capacity_vertex_msg(capacity_margin_proj_vertex, \
                                                                                    ef_pose, "base_link", scaling_factor))


        ### Vertex for capacity margin on the Available Polytope
        ActualposevertexArray_message = self.publish_vertex_pose.publish(create_capacity_vertex_msg(ef_pose, \
                                                                                    array([0,0,0]), "base_link", 1))
        
        ### Vertex for capacity margin on the Available Polytope
        DesiredposevertexArray_message = self.publish_vertex_desired_pose.publish(create_capacity_vertex_msg(self.pos_reference, \
                                                                                    array([0,0,0]), "base_link", 1))
        ### Plane for capacity margin 
        #print('capacity_faces',capacity_faces)

        ### Vertex for capacity margin on the Available Polytope
        CapacitymarginactualArray_message = self.publish_capacity_margin_actual.publish(create_segment_msg(closest_vertex, \
                                            capacity_margin_proj_vertex,ef_pose, "base_link", scaling_factor))
        
        capacityArray_message = self.publish_capacity_margin_polytope.publish(create_polytopes_msg(polytope_verts, capacity_faces, \
                                                                                            ef_pose,"base_link", scaling_factor))


        ########### Estimated Polytope plot ###########################################################################
        
        # Publish polytope faces
        EstpolyArray_message = self.publish_velocity_polytope_est.publish(create_polytopes_msg(polytope_verts_est, polytope_faces_est, \
                                                                                            ef_pose,"base_link", scaling_factor))
        
        

        ### Vertex for capacity margin on the Available Polytope
        EstCapacityprojvertexArray_message = self.publish_vertex_proj_capacity_est.publish(create_capacity_vertex_msg(capacity_margin_proj_vertex_est, \
                                                                                    ef_pose, "base_link", scaling_factor))


        ### Vertex for capacity margin on the Available Polytope
        EstCapacitymarginactualArray_message = self.publish_capacity_margin_actual_est.publish(create_segment_msg(closest_vertex, \
                                            capacity_margin_proj_vertex_est,ef_pose, "base_link", scaling_factor))
        

        EstcapacityArray_message = self.publish_capacity_margin_polytope_est.publish(create_polytopes_msg(polytope_verts_est, capacity_faces_est, \
                                                                                            ef_pose,"base_link", scaling_factor))

        


        ### Vertex 
        
        ##############################################################################################################
        
        
        #print('facet_vertex_idx',facet_vertex_idx)
        
        mutex.release()
        '''
        return_dist_error = norm(pos_act.flatten()-self.pos_reference)



        return float64(return_dist_error)

    def constraint_function(self, q_in):

        #st = time.time()
        # self.joint_state_publisher_robot(q_in)
        #pos_act = array(self.pykdl_util_kin.forward(q_in)[0:3, 3])
        '''
        pin.computeFrameJacobian(self.rmodel, self.rdata,self.q_in_numpy,33)
        pin.forwardKinematics(self.rmodel,self.rdata, self.q_in_numpy)
        pin.updateFramePlacements(self.rmodel,self.rdata)


        pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
        pos_act2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation



        pos_act = pos_act1 + (pos_act2 - pos_act1)*0.5

        
        pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

        
        J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_kuka'), pin.LOCAL_WORLD_ALIGNED)
        J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
        
        J_Hess = hstack((J_Hess1[:,:6],J_Hess2[:,6:]))

        ef_pose = pos_act
        #ef_pose = ef_pose[:, 0]
        # Vertex for capacity margin on the Available Polytope
        ActualposevertexArray_message = self.publish_vertex_pose.publish(create_capacity_vertex_msg(ef_pose,
                                                                                                    array([0, 0, 0]), "base_link", 1))

        # Vertex for capacity margin on the Available Polytope
        DesiredposevertexArray_message = self.publish_vertex_desired_pose.publish(create_capacity_vertex_msg(self.pos_reference,
                                                                                                            array([0, 0, 0]), "base_link", 1))
        #print('distance error norm',distance_error)
        
        J_Hess = array(self.pykdl_util_kin.jacobian(q_in))
        if self.polytope_display:
            scaling_factor = 10.0
            
            mutex.acquire()

            # Polytope plot with estimation

            #pykdl_kin_jac = pykdl_util_kin.jacobian(self.q_in_numpy)
            polytope_verts, polytope_faces, facet_vertex_idx, capacity_faces, capacity_margin_proj_vertex, \
                polytope_verts_est, polytope_faces_est, capacity_faces_est, capacity_margin_proj_vertex_est = \
                velocity_polytope_with_estimation(
                    J_Hess, self.qdot_max, self.qdot_min, self.desired_vertices, self.sigmoid_slope_input)
            desired_polytope_verts, desired_polytope_faces = desired_polytope(
                self.desired_vertices)

            # print('facet_vertex_idx',facet_vertex_idx)
            # print('capacity_margin_proj_vertex',capacity_margin_proj_vertex)
            # print('capacity_margin_proj_vertex_est',capacity_margin_proj_vertex_est)

            # Only for visualization - Polytope at end-effector - No physical significance
            #ef_pose = pykdl_util_kin.forward(self.q_in_numpy)[:3,3]

            # Get end-effector of the robot here for the polytope offset

            #ef_pose = position_70(q_in)

            ef_pose = pos_act
            ef_pose = ef_pose[:, 0]
            #print('ef_pose is',ef_pose)
            #input('stop to test ef_pose')

            ########### Actual POlytope plot ###########################################################################
            # Publish polytope faces
            polyArray_message = self.publish_velocity_polytope.publish(create_polytopes_msg(polytope_verts, polytope_faces,
                                                                                            ef_pose, "base_link", scaling_factor))

            # Desired polytope set - Publish

            DesiredpolyArray_message = self.publish_desired_polytope.publish(create_polytopes_msg(desired_polytope_verts, desired_polytope_faces,
                                                                                                ef_pose, "base_link", scaling_factor))

            # Vertex for capacity margin on the Desired Polytope
            # print('facet_vertex_idx',facet_vertex_idx)
            closest_vertex = self.cartesian_desired_vertices[facet_vertex_idx[0, 1]]
            # print('closest_vertex',closest_vertex)

            CapacityvertexArray_message = self.publish_vertex_capacity.publish(create_capacity_vertex_msg(closest_vertex,
                                                                                                        ef_pose, "base_link", scaling_factor))

            # Vertex for capacity margin on the Available Polytope
            CapacityprojvertexArray_message = self.publish_vertex_proj_capacity.publish(create_capacity_vertex_msg(capacity_margin_proj_vertex,
                                                                                        ef_pose, "base_link", scaling_factor))

            # Vertex for capacity margin on the Available Polytope
            ActualposevertexArray_message = self.publish_vertex_pose.publish(create_capacity_vertex_msg(ef_pose,
                                                                                                        array([0, 0, 0]), "base_link", 1))

            # Vertex for capacity margin on the Available Polytope
            DesiredposevertexArray_message = self.publish_vertex_desired_pose.publish(create_capacity_vertex_msg(self.pos_reference,
                                                                                                                array([0, 0, 0]), "base_link", 1))
            # Plane for capacity margin
            # print('capacity_faces',capacity_faces)

            # Vertex for capacity margin on the Available Polytope
            CapacitymarginactualArray_message = self.publish_capacity_margin_actual.publish(create_segment_msg(closest_vertex,
                                                                                                            capacity_margin_proj_vertex, ef_pose, "base_link", scaling_factor))

            capacityArray_message = self.publish_capacity_margin_polytope.publish(create_polytopes_msg(polytope_verts, capacity_faces,
                                                                                                    ef_pose, "base_link", scaling_factor))

            ########### Estimated Polytope plot ###########################################################################

            # Publish polytope faces
            EstpolyArray_message = self.publish_velocity_polytope_est.publish(create_polytopes_msg(polytope_verts_est, polytope_faces_est,
                                                                                                ef_pose, "base_link", scaling_factor))

            # Vertex for capacity margin on the Available Polytope
            EstCapacityprojvertexArray_message = self.publish_vertex_proj_capacity_est.publish(create_capacity_vertex_msg(capacity_margin_proj_vertex_est,
                                                                                                                        ef_pose, "base_link", scaling_factor))

            # Vertex for capacity margin on the Available Polytope
            EstCapacitymarginactualArray_message = self.publish_capacity_margin_actual_est.publish(create_segment_msg(closest_vertex,
                                                                                                                    capacity_margin_proj_vertex_est, ef_pose, "base_link", scaling_factor))

            EstcapacityArray_message = self.publish_capacity_margin_polytope_est.publish(create_polytopes_msg(polytope_verts_est, capacity_faces_est,
                                                                                                            ef_pose, "base_link", scaling_factor))

            # Vertex

            ##############################################################################################################

            # print('facet_vertex_idx',facet_vertex_idx)

            mutex.release()
        
        #ex_time = time.time() - st
        '''
        self.q_in_numpy[6:] = q_in


        pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)
        #print('self.q_in_numpy is',self.q_in_numpy)
        pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)
        pin.framesForwardKinematics(self.rmodel, self.rdata, self.q_in_numpy)
        #pos_act_int1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
        pos_act_int2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation
        J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
        J = J_Hess2                          
        #oMdes = pin.SE3(eye(3), array([pos_act_int1[0], pos_act_int1[1], pos_act_int1[2]]))
        oMdes = pin.SE3(eye(3), array([self.pos_reference[0], self.pos_reference[1], self.pos_reference[2]]))
        #oMdes = self.pos_reference
        OMcurr = pin.SE3(eye(3), array([pos_act_int2[0], pos_act_int2[1], pos_act_int2[2]]))

        # dMi = oMdes.actInv(OMcurr)
        # err = pin.log(dMi).vector

        # v = - J.T.dot(solve(J.dot(J.T) + damp * eye(6), err))
        # q = pin.integrate(self.rmodel,self.q_in_numpy,v*DT)

        # self.q_in_numpy[6:] = q[6:]
        self.joint_state_publisher_robot(self.q_in_numpy)
        #viz.display(q0)


        return -float64(pos_act_int2 - self.pos_reference)
        # return  norm(pos_act.flatten()-self.pos_reference)

        # Constraints should be actual IK - Actual vs desrired - Cartesian pos

        # NOrm -- || || < 1eps-

    def constraint_function_Gamma(self, q_in):

        #J_Hess = array(self.pykdl_util_kin.jacobian(q_in))
        pin.computeFrameJacobian(self.rmodel, self.rdata,self.q_in_numpy,33)
        pin.forwardKinematics(self.rmodel,self.rdata, self.q_in_numpy)
        pin.updateFramePlacements(self.rmodel,self.rdata)


        #pos_act1 = self.rdata.oMf[self.rmodel.getFrameId('tcp_kuka')].translation
        pos_act2 = self.rdata.oMf[self.rmodel.getFrameId('tcp_meca')].translation



        #pos_act = pos_act1 + (pos_act2 - pos_act1)*0.5

        pos_act = pos_act2

        
        pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

        
        #J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_kuka'), pin.LOCAL_WORLD_ALIGNED)
        J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
        
        #J_Hess = hstack((J_Hess1[:,:6],J_Hess2[:,6:]))
        J_Hess = J_Hess2[:,6:]

        h_plus, h_plus_hat, h_minus, h_minus_hat, p_plus, p_minus, p_plus_hat, p_minus_hat, n_k, Nmatrix, Nnot = get_polytope_hyperplane(
            J_Hess, active_joints=self.active_joints, cartesian_dof_input=array([True, True, True, False, False, False]), qdot_min=self.qdot_min[6:],
            qdot_max=self.qdot_max[6:], cartesian_desired_vertices=self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input)

        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(
            J_Hess, n_k, h_plus, h_plus_hat, h_minus, h_minus_hat,
            active_joints=self.active_joints, cartesian_dof_input=array([True, True, True, False, False, False]), qdot_min=self.qdot_min[6:],
            qdot_max=self.qdot_max[6:], cartesian_desired_vertices=self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input)

        #print('Current objective in optimization Gamma is',self.opt_polytope_model.Gamma_min_softmax)
        return float64(1.0*Gamma_min_softmax)

    def jac_func(self, q_in):

        self.fun_counter += 0.5
        self.fun_iter.data = int(self.fun_counter)
        
        pin.computeJointJacobians(self.rmodel,self.rdata, self.q_in_numpy)

        
        #J_Hess1 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_kuka'), pin.LOCAL_WORLD_ALIGNED)
        J_Hess2 = pin.getFrameJacobian(self.rmodel, self.rdata, self.rmodel.getFrameId('tcp_meca'), pin.LOCAL_WORLD_ALIGNED)
        
        #J_Hess = hstack((J_Hess1[:,:6],J_Hess2[:,6:]))
        J_Hess = J_Hess2[:,6:]

        Hess = getHessian(J_Hess)
        jac_output = mp.Array('f',zeros(shape=(self.active_joints)))


        h_plus, h_plus_hat, h_minus, h_minus_hat, p_plus, p_minus, p_plus_hat, p_minus_hat, n_k, Nmatrix, Nnot = get_polytope_hyperplane(
            J_Hess, active_joints=self.active_joints, cartesian_dof_input=array([True, True, True, False, False, False]), qdot_min=self.qdot_min[6:],
            qdot_max=self.qdot_max[6:], cartesian_desired_vertices=self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input)

        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(
            J_Hess, n_k, h_plus, h_plus_hat, h_minus, h_minus_hat,
            active_joints=self.active_joints, cartesian_dof_input=array([True, True, True, False, False, False]), qdot_min=self.qdot_min[6:],
            qdot_max=self.qdot_max[6:], cartesian_desired_vertices=self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input)

        # self.opt_polytope_gradient_model.compute_polytope_gradient_parameters(self.opt_robot_model,self.opt_polytope_model)
        # self.opt_polytope_gradient_model.Gamma_hat_gradient(sigmoid_slope=1000)
        #st = time.time()
        #jac_output[:] = -10000

        # Create a new thread and start it
        threads = []
        for i_thread in range(self.active_joints):
            thread = mp.Process(target=Gamma_hat_gradient_dq,args=(J_Hess, Hess, n_k, Nmatrix, Nnot, h_plus_hat, h_minus_hat, p_plus_hat,\
                                        p_minus_hat, Gamma_total_hat, Gamma_min_index_hat,\
                                        self.qdot_min[6:], self.qdot_max[6:], self.desired_vertices,self.sigmoid_slope_input,i_thread,jac_output))
            thread.start()
            threads.append(thread)
        
        # now wait for them all to finish
        for thread in threads:
            thread.join()

        '''
        # Method without multi-processing thread
        jac_output[0] = Gamma_hat_gradient_dq(J_Hess, Hess, n_k, Nmatrix, Nnot, h_plus_hat, h_minus_hat, p_plus_hat,
                                        p_minus_hat, Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat,
                                        self.qdot_min, self.qdot_max, self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input,test_joint=0)
        '''

        self.pub_end_ik.publish(self.fun_iter)
        return -float64(jac_output)


if __name__ == '__main__':
    print("KUKA dual arm control start up v1 File\n")
    controller = Geomagic2KUKA()
    # controller.start()
    rospy.spin()