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

#!/usr/bin/env python
## Library import
############# ROS Dependencies #####################################
import rospy
import os
from geometry_msgs import msg
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped,WrenchStamped, PointStamped
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Joy, JointState, PointCloud 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf_conversions as tf_c
from tf2_ros import TransformBroadcaster
import seaborn as sns

#### Polytope Plot - Dependency #####################################

## Polygon plot for ROS - Geometry message
from jsk_recognition_msgs.msg import PolygonArray, SegmentArray
from geometry_msgs.msg import Polygon, PolygonStamped, Point32, Pose


from rospygradientpolytope.visual_polytope import velocity_polytope,desired_polytope,velocity_polytope_with_estimation
from rospygradientpolytope.polytope_ros_message import create_polytopes_msg, create_polygon_msg,create_capacity_vertex_msg, create_segment_msg
from rospygradientpolytope.polytope_functions import get_polytope_hyperplane, get_capacity_margin
from rospygradientpolytope.polytope_gradient_functions import Gamma_hat_gradient
from rospygradientpolytope.sawyer_functions import jacobianE0, position_70
from rospygradientpolytope.robot_functions import getHessian

#################### Linear Algebra ####################################################

from numpy.core.numeric import cross
from numpy import matrix,matmul,transpose,isclose,array,rad2deg,abs,vstack,hstack,shape,eye,zeros,random,savez,load
from numpy import polyfit,poly1d

from numpy.linalg import norm,det
from math import atan2, pi,asin,acos


from re import T

################# URDF Parameter Server ##################################################
# URDF parsing an kinematics - Using pykdl_utils : For faster version maybe use PyKDL directly without wrapper
from urdf_parser_py.urdf import URDF

from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.joint_kinematics import JointKinematics


import scipy.optimize as sco
## Mutex operator
from threading import Lock

# getting the node namespace
#namespace = rospy.get_namespace()
# Import services here

from rospygradientpolytope.srv import IKopt, IKoptResponse


from tf.transformations import quaternion_matrix


mutex = Lock()

# Import URDF of the robot - # todo the param file for fetching the URDF without file location
## Launching the Robot here - Roslaunching an external launch file of the robot


# loading the root urdf from robot_description parameter
# Get the URDF from the parameter server
# Launch the robot 
robot_urdf = URDF.from_parameter_server() 

### For Universal Robot - UR5
base_link = "base_link"

tip_link = "tool0"




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






# maximal joint angles


# Build kinematic chain here

class LaunchRobot():

    def __init__(self):

        

		# publish plytope --- Actual Polytope - Publisher
        
        ## PyKDL_Util here
        
        
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


        # Subscribe joints of Robot --- Joint State subscriber

        self.robot_joint_state_publisher = rospy.Publisher("/joint_states",JointState, queue_size = 1)

        #self.robot_joint_state_subscriber = rospy.Subscriber("/joint_states",JointState,self.joint_state_callback,queue_size=1)

        self.cartesian_desired_vertices = 0.5*array([[0.20000, 0.50000, 0.50000],
                                        [0.50000, -0.10000, 0.50000],
                                        [0.50000, 0.50000, -0.60000],
                                        [0.50000, -0.10000, -0.60000],
                                        [-0.30000, 0.50000, 0.50000],
                                        [-0.30000, -0.10000, 0.50000],
                                        [-0.30000, 0.50000, -0.60000],
                                        [-0.30000, -0.10000, -0.60000]])
        
        
        self.desired_vertices = zeros(shape = (len(self.cartesian_desired_vertices),3))


        self.desired_vertices = self.cartesian_desired_vertices
        
        self.pub_rate = 500 #Hz

        self.sigmoid_slope = 25
        self.robot_joint_names = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']

        self.robot_joint_names_pub = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
        

        # maximal joint velocities
        #self.dq_max = array([pykdl_util_kin.joint_limits_velocity]).T
        #self.dq_min = -self.dq_max
        # maximal joint angles
        #self.q_upper_limit = array([pykdl_util_kin.joint_limits_upper]).T
        #self.q_lower_limit = array([pykdl_util_kin.joint_limits_lower]).T
        
        #self.q_upper_limit = [robot_urdf.joint_map[i].limit.upper - 0.07 for i in self.robot_joint_names]
        #self.q_lower_limit = [robot_urdf.joint_map[i].limit.lower + 0.07 for i in self.robot_joint_names]
        self.qdot_limit = [robot_urdf.joint_map[i].limit.velocity for i in self.robot_joint_names]


        
        self.qdot_max = array(self.qdot_limit)
        self.qdot_min = -1*self.qdot_max

        print('self.qdot_max',self.qdot_max)
        print('self.qdot_min',self.qdot_min)
        self.q_in = zeros(6)
        #self.q_test = zeros(7)

        self.pykdl_util_kin = KDLKinematics(robot_urdf , base_link, tip_link,None)
        #self.q_bounds = zeros(len(self.q_upper_limit),2)

        self.q_upper_limit = array([self.pykdl_util_kin.joint_limits_upper]).T
        #self.q_upper_limit = self.pykdl_util_kin.joint_limits_upper
        self.q_lower_limit = array([self.pykdl_util_kin.joint_limits_lower]).T
        #self.q_lower_limit = self.pykdl_util_kin.joint_limits_lower


        self.q_bounds = hstack((self.q_lower_limit,self.q_upper_limit))

        print('self.q_lower_limit',self.q_lower_limit)
        print('self.q_upper_limit',self.q_upper_limit)
        print('self.q_bounds',self.q_bounds)
        #print('self.q_upper_limit',self.q_upper_limit)
        #print('self.q_lower_limit',self.q_lower_limit)
        print('Velocity limits are', self.qdot_limit)

        '''
        ## First generate a list of random joints in numpy and save in numpy savez array - First generation
        self.q_in_array = zeros(shape = (100,6))
        for i in range(100):
            for j in range(6):
                self.q_in_array[i,j] = random.uniform(self.q_lower_limit[j,0],self.q_upper_limit[j,0])
        
        # Save the file as npz 
        savez('q_in_UR', q_in_arr = self.q_in_array)
        '''

        BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/test_results/"
        test_case = 1
        self.test_case = test_case
        file_name = 'q_in_ur_test_'+str(test_case)

        '''
        ## First generate a list of random joints in numpy and save in numpy savez array - First generation
        self.q_in_array = zeros(shape = (1000,6))
        #for i in range(100):
        for j in range(6):
            self.q_in_array[:,j] = random.uniform(self.q_lower_limit[j,0],self.q_upper_limit[j,0],size=(1000))
        
        # Save the file as npz 
        #savez('q_in_sawyer', q_in_arr = self.q_in_array)
        savez(os.path.join(BASE_PATH, file_name),q_in_arr = self.q_in_array)
        '''
        # Load random joint configurations within the limit
        data_load = load(os.path.join(BASE_PATH, file_name)+str('.npz'))
        #q_in_array_load = load('q_in_sawyer_test_1.npz')
        self.q_in_array = data_load['q_in_arr']


        
        ## Testing gamma vs gamma_hat here
        self.pos_desired = array([[0.49, 0.412, 0.625]])
        sigmoid_slope_test = array([50,100,150,200,400])

        self.sigmoid_slope_test = array([50,100,150,200,400])

        self.Error_gamma_array = zeros(shape=(len(self.q_in_array),len(self.sigmoid_slope_test)))

        self.ts_arr = zeros(shape=(len(self.q_in_array),len(self.sigmoid_slope_test)))

        self.Gamma_min_array =  zeros(shape=(len(self.q_in_array),len(self.sigmoid_slope_test)))

        self.Gamma_min_softmax_array =  zeros(shape=(len(self.q_in_array),len(self.sigmoid_slope_test)))

        self.test_Gamma_vs_Gamma_hat(self.sigmoid_slope_test)

        '''
        for i in range(5):
            self.test_Gamma_vs_Gamma_hat(sigmoid_slope_test[i])
        '''
        
        #self.q_joints_opt = None

        #input('wait here')

        #while not rospy.is_shutdown():
            


        


        '''
        self.pos_desired = array([[0.51, 0.41, 0.55]])
        opt_solution = self.fmin_opt(self.pos_desired,analytical_solver = False)
        #print('self.pos_desired',self.pos_desired)
        print('opt_solution.x',opt_solution.x)
        self.pos_actual_mat = self.pykdl_util_kin.forward(opt_solution.x,tip_link,base_link)
        self.pos_actual = self.pos_actual_mat[0:3,3]

        print('self.pos_actual',self.pos_actual)
        '''

        
        
        '''
        input('stop here')
        self.Gamma_min_softmax = None
        

        #self.q_joints_opt = None

        #opt_solution = self.fmin_opt(self.pos_desired,analytical_solver = True)
        #print('self.pos_desired',self.pos_desired)
        #print('opt_solution.x',opt_solution.x)
        #self.pos_actual_mat = self.pykdl_util_kin.forward(opt_solution.x,tip_link,base_link)
        #self.pos_actual = self.pos_actual_mat[0:3,3]

        #print('self.pos_actual',self.pos_actual)
        
        while not rospy.is_shutdown():
            mutex.acquire()
            msg = JointState()

            msg.name = [self.robot_joint_names[0], self.robot_joint_names[1],self.robot_joint_names[2],self.robot_joint_names[3],
                        self.robot_joint_names[4], self.robot_joint_names[5]]
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'ur_base'
            msg.position = [0.95,0.75,0.70,0.05,0.65,0.05]
            msg.velocity = []
            msg.effort = []
            self.robot_joint_state_publisher.publish(msg)
            #print('msg is',msg)

            ## Creating ROS Service for Inverse kinematics Optimization based on Capacity Margin Constraint

            #self.ik_opt_serv = rospy.Service('ik_optimization',pos_des,self.fmin_opt)
            rate = rospy.Rate(self.pub_rate) # Hz
            rate.sleep()
            mutex.release()
        '''
    def joint_state_callback(self,robot_joints):

        ## Get Joint angles of the Sawyer Robot
        self.q_in[0] = robot_joints.position[0]
        self.q_in[1] = robot_joints.position[1]
        self.q_in[2] = robot_joints.position[2]
        self.q_in[3] = robot_joints.position[3]
        self.q_in[4] = robot_joints.position[4]
        self.q_in[5] = robot_joints.position[5]

        

        #self.q_in[3] = 0.5
        #self.q_in[4] = 0.89

        print('self.q_in',self.q_in)
        print('tip_link',tip_link)
        print('base_link',base_link)
        ef = self.pykdl_util_kin.forward(self.q_in,tip_link,base_link)

    
    #def check_gradient(self,q_in,step_size:int):
    def joint_state_publisher_robot(self):
        

        
        q_in = self.q_in
        msg = JointState()

        msg.name = [self.robot_joint_names_pub[0], self.robot_joint_names_pub[1],self.robot_joint_names_pub[2],self.robot_joint_names_pub[3],
                    self.robot_joint_names_pub[4], self.robot_joint_names_pub[5]]
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'ur_base'
        msg.position = [q_in[0],q_in[1],q_in[2],q_in[3],q_in[4],q_in[5]]
        msg.velocity = []
        msg.effort = []        
        self.robot_joint_state_publisher.publish(msg)



    def test_Gamma_vs_Gamma_hat(self,sigmoid_slope_test):
    
        import time
        from numpy.linalg import det
        from numpy import sum,mean,average
        import matplotlib.pyplot as plt

        #from numpy.random import randn
        #from numpy.linalg import norm
        #from linearalgebra import V_unit
        #from copy import deepcopy
        #from numpy import amax,max,exp,dot,matmul,ones
        #from polytope_functions import get_hyperplane_parameters,get_gamma,get_gamma_hat
        #from robot_functions import exp_sum,exp_normalize,smooth_max_gradient
        #from numpy import unravel_index,argmax,min

        #from numpy.linalg import det
        #from numpy import sum

        

        

        #print('q_in is',q_in)
        #input('wait here in the objective function')
        #pos_act_mat = self.pykdl_util_kin.forward(q0,tip_link,base_link)
        #pos_act = pos_act_mat[0:3,3]
        #print('self.pos_reference',self.pos_reference)
        #print('Current position in optimization is',pos_act)
        #print('pos_act is this in the objective function',pos_act)
        #input('inside obj func')
        #self.canvas_input_opt.generate_axis()
        #self.opt_robot_model.urdf_transform(q_joints=q_des)
        #canvas_input.generate_axis()
        BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/test_results/"
        #test_case = 1
        #file_name = 'q_arr_sawyer_test_'+str(test_case)
        
        #full_name = os.path.join(BASE_PATH, file_name)
        

        #Gamma_min_array = zeros(shape = len(self.q_in_array))
        #Gamma_min_softmax_arr = zeros(shape = len(self.q_in_array))
        #Error_arr = zeros(shape = len(self.q_in_array))
        #ts_arr = zeros(shape = len(self.q_in_array))

        Gamma_min_prev = None

        test_joint = 0
        q_add = zeros(6)
        #q_add[test_joint] = 1
        step_size = 0.001

        ## 52,47, 33 test case works perfectly

        #q_in = self.q_in_array[33,:]
        '''
        q_in = q_add
        q_in[0] = 0.50
        q_in[1] = 0.25
        q_in[2] = -0.46
        q_in[3] = -0.75
        q_in[4] = 0.75
        q_in[5] = 1.50
        q_in[6] = -0.76
        '''
        #q_in = self.q_in_array[57,:]
        for j in range(0,len(sigmoid_slope_test)):
            sigmoid_slope_inp = sigmoid_slope_test[j]
            print('completed one test case')
            for i in range(len(self.q_in_array)):
                start_time = time.time()
                #q_in[test_joint] += step_size
                q_in = self.q_in_array[i,:]
                self.q_in = q_in

                #print('self.q_in',q_in)

                self.joint_state_publisher_robot()
                #time.sleep(0.5)

                # Publish joint sates to Rviz for the robot
                

                #J_Hess = jacobianE0(q_in) 
                #ef_pose = self.pykdl_util_kin.forward(q_in)[:3,3]
                J_Hess = array(self.pykdl_util_kin.jacobian(q_in,pos=None))


                #q_in = 
                #J_Hess_pykdl = array(self.pykdl_util_kin.jacobian(q_in))
                #J_Hess = J_Hess[0:3,:]
                ##print('J_Hess',J_Hess)
                #print('J_Hess_pykdl',J_Hess_pykdl)
                #input('wait here')
                h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
                    J_Hess,active_joints=6,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
                    qdot_max=self.qdot_max,cartesian_desired_vertices= self.cartesian_desired_vertices,sigmoid_slope=sigmoid_slope_inp )


                Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
                    J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                                active_joints=6,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
                    qdot_max=self.qdot_max,cartesian_desired_vertices= self.cartesian_desired_vertices,sigmoid_slope=sigmoid_slope_inp )
                #self.Gamma_min_array[i] = Gamma_min
                #Gamma_min_softmax_arr[i] = Gamma_min_softmax




            
                scaling_factor = 10.0

                '''    
                ### Actual polytope plot
                polytope_verts, polytope_faces, facet_vertex_idx, capacity_faces, capacity_margin_proj_vertex = \
                                                velocity_polytope(pykdl_kin_jac,self.qdot_limit,-1*self.qdot_limit)
                '''
                
                ### Polytope plot with estimation
                '''
                #pykdl_kin_jac = pykdl_util_kin.jacobian(self.q_in_numpy)
                polytope_verts, polytope_faces, facet_vertex_idx, capacity_faces, capacity_margin_proj_vertex, \
                    polytope_verts_est, polytope_faces_est, capacity_faces_est, capacity_margin_proj_vertex_est = \
                                                velocity_polytope_with_estimation(J_Hess,self.qdot_max,self.qdot_min,self.cartesian_desired_vertices,sigmoid_slope_inp)
                desired_polytope_verts, desired_polytope_faces = desired_polytope(self.cartesian_desired_vertices)

                #print('facet_vertex_idx',facet_vertex_idx)
                #print('capacity_margin_proj_vertex',capacity_margin_proj_vertex)
                #print('capacity_margin_proj_vertex_est',capacity_margin_proj_vertex_est)
                

                # Only for visualization - Polytope at end-effector - No physical significance
                #ef_pose = pykdl_util_kin.forward(self.q_in_numpy)[:3,3]

                mutex.acquire()
                

                # Get end-effector of the robot here for the polytope offset

                #ef_pose = position_70(q_in)

                ef_pose = self.pykdl_util_kin.forward(q_in)[:3,3]

                ########### Actual POlytope plot ###########################################################################
                # Publish polytope faces
                polyArray_message = self.publish_velocity_polytope.publish(create_polytopes_msg(polytope_verts, polytope_faces, \
                                                                                                    ef_pose,"right_arm_base_link", scaling_factor))
                
                
                ### Desired polytope set - Publish

                DesiredpolyArray_message = self.publish_desired_polytope.publish(create_polytopes_msg(desired_polytope_verts, desired_polytope_faces, \
                                                                                                    ef_pose,"right_arm_base_link", scaling_factor))


                ### Vertex for capacity margin on the Desired Polytope
                #print('facet_vertex_idx',facet_vertex_idx)
                closest_vertex = self.cartesian_desired_vertices[facet_vertex_idx[0,1]]
                #print('closest_vertex',closest_vertex)

                CapacityvertexArray_message = self.publish_vertex_capacity.publish(create_capacity_vertex_msg(closest_vertex, \
                                                                                            ef_pose, "right_arm_base_link", scaling_factor))
                
                ### Vertex for capacity margin on the Available Polytope
                CapacityprojvertexArray_message = self.publish_vertex_proj_capacity.publish(create_capacity_vertex_msg(capacity_margin_proj_vertex, \
                                                                                            ef_pose, "right_arm_base_link", scaling_factor))
                ### Plane for capacity margin 
                #print('capacity_faces',capacity_faces)

                ### Vertex for capacity margin on the Available Polytope
                CapacitymarginactualArray_message = self.publish_capacity_margin_actual.publish(create_segment_msg(closest_vertex, \
                                                    capacity_margin_proj_vertex,ef_pose, "right_arm_base_link", scaling_factor))
                
                capacityArray_message = self.publish_capacity_margin_polytope.publish(create_polytopes_msg(polytope_verts, capacity_faces, \
                                                                                                    ef_pose,"right_arm_base_link", scaling_factor))


                ########### Estimated Polytope plot ###########################################################################
                
                # Publish polytope faces
                EstpolyArray_message = self.publish_velocity_polytope_est.publish(create_polytopes_msg(polytope_verts_est, polytope_faces_est, \
                                                                                                    ef_pose,"right_arm_base_link", scaling_factor))
                
                

                ### Vertex for capacity margin on the Available Polytope
                EstCapacityprojvertexArray_message = self.publish_vertex_proj_capacity_est.publish(create_capacity_vertex_msg(capacity_margin_proj_vertex_est, \
                                                                                            ef_pose, "right_arm_base_link", scaling_factor))


                ### Vertex for capacity margin on the Available Polytope
                EstCapacitymarginactualArray_message = self.publish_capacity_margin_actual_est.publish(create_segment_msg(closest_vertex, \
                                                    capacity_margin_proj_vertex_est,ef_pose, "right_arm_base_link", scaling_factor))
                

                EstcapacityArray_message = self.publish_capacity_margin_polytope_est.publish(create_polytopes_msg(polytope_verts_est, capacity_faces_est, \
                                                                                                    ef_pose,"right_arm_base_link", scaling_factor))

                
                ### Vertex 
                
                ##############################################################################################################
                mutex.release()
                '''
                #Hess = getHessian(J_Hess)
                #print('Gamma_min_softmax',Gamma_min_softmax)
                '''
                jac_output = Gamma_hat_gradient(J_Hess,Hess,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                    p_minus_hat,Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat,\
                    self.qdot_min,self.qdot_max,self.desired_vertices,sigmoid_slope=sigmoid_slope_inp)
                
                if Gamma_min_prev != None:

                    #for i in range(0,limit,step_size):
                    numerical_gradient = (Gamma_min_prev - Gamma_min_softmax)/(step_size*1.0)

                    print('numerical_gradient',numerical_gradient)
                    print('Jacobian output is',jac_output)
                    analytical_gradient = jac_output[test_joint+1
                    ]
                    print('analytical_gradient',analytical_gradient)

                    error_grad = ((analytical_gradient - numerical_gradient)/(numerical_gradient*1.0))*100.0

                    
                    print('error is',error_grad)
                Gamma_min_prev = Gamma_min_softmax
                '''
                
                self.Error_gamma_array[i,j] = (Gamma_min - Gamma_min_softmax)#/(1.0*Gamma_min))*100 
                

                self.Gamma_min_array[i,j] = Gamma_min

                self.Gamma_min_softmax_array[i,j] = Gamma_min_softmax

                self.ts_arr[i,j] = (time.time() - start_time)


                #self.Error_arr[i] = ((Gamma_min_softmax - Gamma_min)/(1.0*Gamma_min))*100 
                #ts_arr[i] = (time.time() - start_time)

                


                '''

                file_name = 'sawyer_test_Gamma_vs_Gamma_hat_slope_'+str(sigmoid_slope_inp) + str('_') + str(i)
            
                savez(os.path.join(BASE_PATH, file_name),q_in = self.q_in_array,sigmoid_slope_inp = sigmoid_slope_inp, ts = self.ts_arr[i,j],\
                    h_plus = h_plus, h_plus_hat = h_plus_hat, h_minus = h_minus, h_minus_hat = h_minus_hat,\
                Gamma_minus = Gamma_minus, Gamma_plus = Gamma_plus, Gamma_total_hat = Gamma_total_hat, Gamma_min = Gamma_min, Gamma_min_softmax = Gamma_min_softmax, \
                    Gamma_min_index_hat =  Gamma_min_index_hat, facet_pair_idx = facet_pair_idx, hyper_plane_sign = hyper_plane_sign)
                '''    
        print('completed one test case')
        print('Gamma_min_arr_Mean is:',mean(self.Gamma_min_array))
        print('Gamma_min_arr_average is:',average(self.Gamma_min_array))
        print('Gamma_min_softmax Mean is',mean(self.Gamma_min_softmax_array))
        print('Gamma_min_softmax Average is',average(self.Gamma_min_softmax_array))

        print('sigmoid_slope is',sigmoid_slope_inp)
        print('Mean error is:',mean(self.Error_gamma_array))
        print('Average error is:',average(self.Error_gamma_array))
        print('Average time to execute',average(self.ts_arr))

        fig, ax = plt.subplots()

        max_gamma = zeros(shape=(len(sigmoid_slope_test)))

        ## Get the maximum value of the Capacity margin to normalize the vector of error
        for i in range(len(sigmoid_slope_test)):
            max_gamma[i] = max(self.Gamma_min_array[:,i])
        
        data_to_plot = [self.Error_gamma_array[:,0]/(1.0*max_gamma[0]),self.Error_gamma_array[:,1]/(1.0*max_gamma[1]),\
                        self.Error_gamma_array[:,2]/(1.0*max_gamma[2]),self.Error_gamma_array[:,3]/(1.0*max_gamma[3]),self.Error_gamma_array[:,4]/(1.0*max_gamma[4])]

        # List of labels from sigmoid slope
        ax.set_xticklabels(['','50','100','150','200','400'])
        plt.xlabel("Sigmoid Slope",size=15)
        plt.ylabel("Error",size=15)
        # Create the boxplot
        bp = ax.violinplot(data_to_plot)
        #ax.set_aspect(1)
        
        plt.savefig('Error_plot_UR_' + str(self.test_case)+('.png'))
        plt.show()

        fig2, ax2 = plt.subplots()
        # List of five airlines to plot
        sigmoid_slope_plot = ['50', '100', '150',
                '200', '400']

        # Iterate through the five airlines
        '''

        for i in range(len(sigmoid_slope_plot)):
            # Subset to the airline
            subset = flights[flights['name'] == airline]
            
            # Draw the density plot
            sns.distplot(subset['arr_delay'], hist = False, kde = True,
                        kde_kws = {'linewidth': 3},
                        label = sigmoid_slope_plot[i])
        plt.legend(prop={'size': 16}, title = 'Sigmoid Slope')
        #plt.title('Density Plot with Multiple Airlines')
        plt.xlabel('Error (%)')
        plt.ylabel('Capacity Margin')

        sns.distplot(subset['arr_delay'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = Sigmoid Slope)
            
        '''
        # Plot formatting

        # Plot formatting

        ax2.set_ylim([-0.02, 0.12])
        #self.Gamma_min_array[:,0] = self.Gamma_min_array[:,0]

        plt_sigmoid_50 = plt.scatter(self.Gamma_min_array[:,0],self.Error_gamma_array[:,0]/(1.0*max_gamma[0]),color='c',s=0.5,alpha=0.5)

        #calculate equation for trendline
        z_50 = polyfit(self.Gamma_min_array[:,0], self.Error_gamma_array[:,0]/(1.0*max_gamma[0]), 1)
        p_50 = poly1d(z_50)

        #add trendline to plot
        plt.plot(self.Gamma_min_array[:,0], p_50(self.Gamma_min_array[:,0]),color='c')   

        plt_sigmoid_100 = plt.scatter(self.Gamma_min_array[:,1],self.Error_gamma_array[:,1]/(1.0*max_gamma[1]),color='m',s=0.5,alpha=0.5)

        #calculate equation for trendline
        z_100 = polyfit(self.Gamma_min_array[:,1], self.Error_gamma_array[:,1]/(1.0*max_gamma[1]), 1)
        p_100 = poly1d(z_100)

        #add trendline to plot
        plt.plot(self.Gamma_min_array[:,1], p_100(self.Gamma_min_array[:,1]),color='m')   


        plt_sigmoid_150 = plt.scatter(self.Gamma_min_array[:,2],self.Error_gamma_array[:,2]/(1.0*max_gamma[2]),color='y',s=0.5,alpha=0.5)


        #calculate equation for trendline
        z_150 = polyfit(self.Gamma_min_array[:,2], self.Error_gamma_array[:,2]/(1.0*max_gamma[2]), 1)
        p_150 = poly1d(z_150)

        #add trendline to plot
        plt.plot(self.Gamma_min_array[:,2], p_150(self.Gamma_min_array[:,2]),color='y')   

        plt_sigmoid_200 = plt.scatter(self.Gamma_min_array[:,3],self.Error_gamma_array[:,3]/(1.0*max_gamma[3]),color='k',s=0.5,alpha=0.5)


        #calculate equation for trendline
        z_200 = polyfit(self.Gamma_min_array[:,3], self.Error_gamma_array[:,3]/(1.0*max_gamma[3]), 1)
        p_200 = poly1d(z_200)

        #add trendline to plot
        plt.plot(self.Gamma_min_array[:,3], p_200(self.Gamma_min_array[:,3]),color='k')   

        plt_sigmoid_400 = plt.scatter(self.Gamma_min_array[:,4],self.Error_gamma_array[:,4]/(1.0*max_gamma[4]),color='r',s=0.5,alpha=0.5)
        #plt.scatter(plt_sigmoid_50_x,plt_sigmoid_50_y,'r')


        #calculate equation for trendline
        z_400 = polyfit(self.Gamma_min_array[:,4], self.Error_gamma_array[:,4]/(1.0*max_gamma[4]), 1)
        p_400 = poly1d(z_400)

        #add trendline to plot
        plt.plot(self.Gamma_min_array[:,4], p_400(self.Gamma_min_array[:,4]),color='r')  

        plt.legend((plt_sigmoid_50 ,plt_sigmoid_100 ,plt_sigmoid_150 ,plt_sigmoid_200, plt_sigmoid_400  ),('50','100', '150','200','400'),title='Sigmoid Slope',markerscale=5)
        plt.xlabel('Actual Capacity Margin (m/s)',size=15)
        plt.ylabel('Error',size = 15,labelpad=-6)
        #self.Gamma_min_array

        #self.Gamma_min_array

        

        plt.show()






        file_name = 'error_plot_UR_'+str(self.test_case)

        savez(os.path.join(BASE_PATH, file_name),q_in = self.q_in_array,sigmoid_slope_inp = sigmoid_slope_test, ts = self.ts_arr,\
            Gamma_min_array = self.Gamma_min_array, Gamma_min_softmax_array = self.Gamma_min_softmax_array,Error_gamma_array = self.Error_gamma_array)

    def test_Gamma_vs_Gamma_hat_old(self,sigmoid_slope_test):
        
        import time
        from numpy.linalg import det
        from numpy import sum,mean,average
        #from numpy.random import randn
        #from numpy.linalg import norm
        #from linearalgebra import V_unit
        #from copy import deepcopy
        #from numpy import amax,max,exp,dot,matmul,ones
        #from polytope_functions import get_hyperplane_parameters,get_gamma,get_gamma_hat
        #from robot_functions import exp_sum,exp_normalize,smooth_max_gradient
        #from numpy import unravel_index,argmax,min

        #from numpy.linalg import det
        #from numpy import sum

        sigmoid_slope_inp = sigmoid_slope_test

        

        #print('q_in is',q_in)
        #input('wait here in the objective function')
        #pos_act_mat = self.pykdl_util_kin.forward(q0,tip_link,base_link)
        #pos_act = pos_act_mat[0:3,3]
        #print('self.pos_reference',self.pos_reference)
        #print('Current position in optimization is',pos_act)
        #print('pos_act is this in the objective function',pos_act)
        #input('inside obj func')
        #self.canvas_input_opt.generate_axis()
        #self.opt_robot_model.urdf_transform(q_joints=q_des)
        #canvas_input.generate_axis()
        BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/test_results/"
        
        
        #full_name = os.path.join(BASE_PATH, file_name)
        print('completed one test case')

        Gamma_min_array = zeros(shape = len(self.q_in_array))
        Gamma_min_softmax_arr = zeros(shape = len(self.q_in_array))
        Error_arr = zeros(shape = len(self.q_in_array))
        ts_arr = zeros(shape = len(self.q_in_array))


        Gamma_min_prev = None

        
        #Awesome configuration
        test_joint = 4
        q_add = zeros(6)
        #q_add[test_joint] = 1
        step_size = 0.001

        ## 52,47, 33 test case works perfectly
        # self.q_in_array[i,:]
        #q_in = self.q_in_array[33,:]

        '''
        ### COnfiguration for the picture in the paper - Capacity Margin vs Estimate 
        ## Sigmoid slope = 10
        q_in = q_add
        q_in[0] = 0.0
        q_in[1] = -1.95
        q_in[2] = 1.35
        q_in[3] = -1.50
        q_in[4] = -1.50
        q_in[5] = 0.0
        
        ### COnfiguration for the picture in the paper- Capacity margin overview picture
        q_in = q_add
        q_in[0] = 0.0
        q_in[1] = -1.95
        q_in[2] = 1.25
        q_in[3] = -1.25
        q_in[4] = -1.60
        q_in[5] = 0.0

        
        #Awesome configuration
        test_joint = 3
        q_add = zeros(6)
        #q_add[test_joint] = 1
        step_size = 0.001

        ## 52,47, 33 test case works perfectly
        # self.q_in_array[i,:]
        #q_in = self.q_in_array[33,:]
        q_in = q_add
        q_in[0] = -0.95
        q_in[1] = -0.95
        q_in[2] = -0.66
        q_in[3] = -0.15
        q_in[4] = 0.12
        q_in[5] = -0.05
        

        
        #Awesome configuration
        test_joint = 3
        q_add = zeros(6)
        #q_add[test_joint] = 1
        step_size = 0.0001

        ## 52,47, 33 test case works perfectly
        # self.q_in_array[i,:]
        #q_in = self.q_in_array[33,:]
        q_in = q_add
        q_in[0] = 1.95
        q_in[1] = 0.95
        q_in[2] = 0.66
        q_in[3] = 0.15
        q_in[4] = -0.12
        q_in[5] = -0.05
        '''
        for i in range(len(self.q_in_array)):
            start_time = time.time()
            #J_Hess = jacobianE0(self.q_in_array[i,:])   
            # Offset set to <xyz> from <-0.01,0,0> to <-0.03,-0.03,0.05>
            # <origin xyz="0.05 ${wrist_3_length} 0.05" rpy="0.0 0.0 ${pi/2.0}" /> 
            # from <origin xyz="0 ${wrist_3_length} 0" rpy="0.0 0.0 ${pi/2.0}" />
            
            mutex.acquire()
            q_in[test_joint] += step_size

            self.q_in = q_in

            print('self.q_in',q_in)

            self.joint_state_publisher_robot()

            ef_pose = self.pykdl_util_kin.forward(self.q_in)[:3,3]
            J_Hess = array(self.pykdl_util_kin.jacobian(self.q_in))
            Hess = getHessian(J_Hess)
            #print('J_Hess',J_Hess)
            #J_Hess = J_Hess[0:3,:]
            ##print('J_Hess',J_Hess)
            #print('J_Hess_pykdl',J_Hess_pykdl)
            #input('wait here')
            h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
                J_Hess,active_joints=6,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
                qdot_max=self.qdot_max,cartesian_desired_vertices= self.cartesian_desired_vertices,sigmoid_slope=sigmoid_slope_inp )


            Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
                J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                            active_joints=6,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
                qdot_max=self.qdot_max,cartesian_desired_vertices= self.cartesian_desired_vertices,sigmoid_slope=sigmoid_slope_inp )
            

            print('Gamma_min_softmax',Gamma_min_softmax)

            print('Jacobian outside Gamma_hat_gradient',J_Hess)
            input('Jacobian outside Gamma_hat_gradient')
    
            jac_output = Gamma_hat_gradient(J_Hess,Hess,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                p_minus_hat,Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat,\
                self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,sigmoid_slope=sigmoid_slope_inp)

            if Gamma_min_prev != None:

                #for i in range(0,limit,step_size):
                numerical_gradient = (Gamma_min_prev - Gamma_min_softmax) /(step_size*1.0)

                print('numerical_gradient',numerical_gradient)

                analytical_gradient = jac_output[test_joint]
                print('analytical_gradient',analytical_gradient)

                error_grad = ((analytical_gradient - numerical_gradient)/(numerical_gradient*1.0))*100.0
                print('self.q_in',q_in)
                print('Gradient is',jac_output)
                print('error is',error_grad)

                print('facet_pair_idx',facet_pair_idx)
                input('stop to test here')
            Gamma_min_prev = Gamma_min_softmax
            

            '''    
            ### Actual polytope plot
            polytope_verts, polytope_faces, facet_vertex_idx, capacity_faces, capacity_margin_proj_vertex = \
                                            velocity_polytope(pykdl_kin_jac,self.qdot_limit,-1*self.qdot_limit)
            '''

            scaling_factor = 10.0
            
            ### Polytope plot with estimation
            
            #pykdl_kin_jac = pykdl_util_kin.jacobian(self.q_in_numpy)
            polytope_verts, polytope_faces, facet_vertex_idx, capacity_faces, capacity_margin_proj_vertex, \
                polytope_verts_est, polytope_faces_est, capacity_faces_est, capacity_margin_proj_vertex_est = \
                                            velocity_polytope_with_estimation(J_Hess,self.qdot_max,self.qdot_min,self.cartesian_desired_vertices,sigmoid_slope_inp)
            desired_polytope_verts, desired_polytope_faces = desired_polytope(self.cartesian_desired_vertices)



            #print('facet_vertex_idx',facet_vertex_idx)
            #print('capacity_margin_proj_vertex',capacity_margin_proj_vertex)
            #print('capacity_margin_proj_vertex_est',capacity_margin_proj_vertex_est)
            

            # Only for visualization - Polytope at end-effector - No physical significance
            #ef_pose = pykdl_util_kin.forward(self.q_in_numpy)[:3,3]

            
            

            # Get end-effector of the robot here for the polytope offset

            #ef_pose = position_70(q_in)

            ef_pose = self.pykdl_util_kin.forward(self.q_in)[:3,3]

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
            
            
            print('facet_vertex_idx',facet_vertex_idx)
            
            mutex.release()
            
            Error_arr[i] = ((Gamma_min_softmax - Gamma_min)/(1.0*Gamma_min))*100 
            ts_arr[i] = (time.time() - start_time)
            
            Gamma_min_array[i] = Gamma_min
            Gamma_min_softmax_arr[i] = Gamma_min_softmax
            #Error_arr[i] = ((Gamma_min_softmax - Gamma_min)/(1.0*Gamma_min))*100 
            #ts_arr[i] = (time.time() - start_time)
            

            '''
            file_name = 'UR_test_Gamma_vs_Gamma_hat_slope_'+str(sigmoid_slope_inp) + str('_') + str(i)
        
            savez(os.path.join(BASE_PATH, file_name),q_in = self.q_in_array,sigmoid_slope_inp = sigmoid_slope_inp, ts = ts_arr[i],\
                h_plus = h_plus, h_plus_hat = h_plus_hat, h_minus = h_minus, h_minus_hat = h_minus_hat,\
            Gamma_minus = Gamma_minus, Gamma_plus = Gamma_plus, Gamma_total_hat = Gamma_total_hat, Gamma_min = Gamma_min, Gamma_min_softmax = Gamma_min_softmax, \
                Gamma_min_index_hat =  Gamma_min_index_hat, facet_pair_idx = facet_pair_idx, hyper_plane_sign = hyper_plane_sign)
            '''    
        print('completed one test case')
        print('Gamma_min_arr_Mean is:',mean(Gamma_min_array))
        print('Gamma_min_arr_average is:',average(Gamma_min_array))
        print('Gamma_min_softmax Mean is',mean(Gamma_min_softmax_arr))
        print('Gamma_min_softmax Average is',average(Gamma_min_softmax_arr))

        print('sigmoid_slope is',sigmoid_slope_inp)
        print('Mean error is:',mean(Error_arr))
        print('Average error is:',average(Error_arr))
        print('Average time to execute',average(ts_arr))
        


    def fmin_opt(self,pose_desired,analytical_solver:bool):
        ### Function - func
        ## Initial point - x0
        ## args -
        ## method - SLQSQ
        ## jac = Jacobian - gradient of the


        #self.opt_polytope_model = polytope_model
        #self.opt_polytope_gradient_model = polytope_gradient_model

        #self.q_joints_input = robot_model.q_joints




        #self.obj_function = polytope_model.Gamma_total
        #print('self.obj_function',self.obj_function)

        self.initial_x0 = self.q_in


        #self.initial_x0 = randn(6)
        #print('self.initial_x0 ',self.initial_x0 )
        #print('self.obj_function(robot_model.q_joints)',self.obj_function(robot_model.q_joints))
        #self.func_deriv = polytope_gradient_model.d_gamma_hat


        #self.opt_polytope_gradient_model.compute_polytope_gradient_parameters(self.opt_robot_model,self.opt_polytope_model)
        #self.opt_polytope_gradient_model.Gamma_hat_gradient(sigmoid_slope=self.sigmoid_slope)
        #methods = trust-constr

        #x0 = self.initial_x0
        #x0_d = self.initial_x0 + 1e-5
        #numerical_err = (self.obj_function(x0_d) - self.obj_function(x0))
        #grad_err = (self.jac_func(x0_d) - self.jac_func(x0))


        #print('numerical is:',numerical_err)
        #print('analytical error is',grad_err)
        #assert sco.check_grad(func = self.obj_function, grad = self.jac_func, x0 = self.initial_x0,espilon = 1e-5, direction = 'all',seed = None)
        ### Get the end-effector posision
        #self.pos_reference = self.opt_robot_model.end_effector_position

        self.pos_reference = pose_desired
        #print('Reference position is',self.pos_reference)


        ### Desired vertex set

        self.desired_vertices[:,0] = self.cartesian_desired_vertices[:,0] # + self.pos_reference[0,0]
        self.desired_vertices[:,1] = self.cartesian_desired_vertices[:,1] #+ self.pos_reference[0,1]
        self.desired_vertices[:,2] = self.cartesian_desired_vertices[:,2] #+ self.pos_reference[0,2]
        #print('self.opt_polytope_gradient_model.d_gamma_hat',self.opt_polytope_gradient_model.d_gamma_hat)

        # Bounds created from the robot angles

        self.opt_bounds = self.q_bounds

        print('self.opt_bounds is',self.opt_bounds)

        ### Constraints

        
        cons = ({'type': 'ineq', 'fun': self.constraint_func},\
                {'type': 'ineq', 'fun': self.constraint_func_Gamma})
        '''
        cons = ({'type': 'ineq', 'fun': self.constraint_func})
        '''





        '''
        cons = ({'type': 'ineq', 'fun': lambda x:  self.q_joints_input[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
        '''
        ## jACOBIAN MATRIX OF THE OBJECTIVE FUNCTION

        # jac = self.opt_polytope_gradient_model.d_gamma_hat,

        ### WIth analytical gradient

        '''
        
                    self.q_joints_opt = sco.minimize(fun = self.obj_function,  x0 = self.initial_x0,bounds = self.opt_bounds,\
                                         jac =self.jac_func,method='SLSQP', \
                                             options={'disp': True,'maxiter':500})
        
        '''

        if analytical_solver:
            q_joints_opt = sco.minimize(fun = self.obj_function,  x0 = self.initial_x0,bounds = self.opt_bounds,\
                                         jac = self.jac_func, constraints=cons,method='SLSQP',\
                                             options={'disp': True,'maxiter':500})
        else:

            q_joints_opt = sco.minimize(fun = self.obj_function,  x0 = self.initial_x0,bounds = self.opt_bounds,\
                                             constraints = cons,tol=1e-6,method='COBYLA', \
                                                 options={'disp': True})




        


        return q_joints_opt
        #hess = self.hess_func,
        #sco.check_grad(func = self.obj_function, grad =self.opt_polytope_gradient_model.d_gamma_hat \                                  , x0 = self.initial_x0, epsilon=1.4901161193847656e-08, direction='all', seed=None)
        #def constraint2(self):
    ## Obj
    def obj_function(self,q_in):



        from numpy.linalg import det
        from numpy import sum


        print('q_in is',q_in)
        #input('wait here in the objective function')
        #pos_act_mat = self.pykdl_util_kin.forward(q0,tip_link,base_link)
        #pos_act = pos_act_mat[0:3,3]
        #print('self.pos_reference',self.pos_reference)
        #print('Current position in optimization is',pos_act)
        #print('pos_act is this in the objective function',pos_act)
        #input('inside obj func')
        #self.canvas_input_opt.generate_axis()
        #self.opt_robot_model.urdf_transform(q_joints=q_des)
        #canvas_input.generate_axis()
        #J_Hess = jacobianE0(q_in)
        #J_Hess_pykdl = array(self.pykdl_util_kin.jacobian(q_in))

        J_Hess = array(self.pykdl_util_kin.jacobian(q_in))
        #J_Hess = J_Hess[0:3,:]
        ##print('J_Hess',J_Hess)
        #print('J_Hess_pykdl',J_Hess_pykdl)
        #input('wait here')
        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
            J_Hess,active_joints=6,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)


        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
            J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints=6,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)
        
        self.Gamma_min_softmax = Gamma_min_softmax
        #print('Current objective function in optimization is',-self.opt_polytope_model.Gamma_min_softmax)
        print('Current objective function in optimization is',-1.0*Gamma_min_softmax)
        

        #err_pos = abs((norm(pos_act-self.pos_reference) - 1e-6))
        #print('error',err_pos)
        #return err_pos
        return -1.0*self.Gamma_min_softmax


    def constraint_func(self,q_in):

        pos_act_mat = self.pykdl_util_kin.forward(q_in,tip_link,base_link)
        
        pos_act = pos_act_mat[0:3,3]

        #pos_act = position_70(q_in)
        print('self.pos_reference',self.pos_reference)
        print('Current position in optimization is',pos_act)
        #input('Wait here ')
        return ( 1e-3 - norm(pos_act-self.pos_reference))



    ### Constraints should be actual IK - Actual vs desrired - Cartesian pos

    ## NOrm -- || || < 1eps-
    def constraint_func_Gamma(self,q_in):

        #self.opt_robot_model.urdf_transform(q_joints=q_des)
        #canvas_input.generate_axis()
        
        J_Hess = array(self.pykdl_util_kin.jacobian(q_in))

        #J_Hess = jacobianE0(q_in)
        #J_Hess = J_Hess[0:3,:]

        #print('self.qdot_min',-1*self.qdot_max)
        #print('self.qdot_max',self.qdot_max)
        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
            J_Hess,active_joints=6,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)


        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
            J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints=6,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)
        
        #print('Current objective in optimization Gamma is',self.opt_polytope_model.Gamma_min_softmax)
        return Gamma_min_softmax

    def jac_func(self,q_in):
        from numpy import sum

        J_Hess = array(self.pykdl_util_kin.jacobian(q_in))
        #J_Hess = jacobianE0(q_in)
        #J_Hess = J_Hess[0:3,:]
        #print('J_Hess',J_Hess)
        Hess = getHessian(J_Hess)

        #print('Hessian',Hess)

        #input('wait here for hessian')
        #J_Hess = J_Hess[0:3,:]

        #print('self.qdot_min',-1*self.qdot_max)
        #print('self.qdot_max',self.qdot_max)
        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(\
            J_Hess,active_joints=6,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)


        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
            J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints=6,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)


        #self.opt_polytope_gradient_model.compute_polytope_gradient_parameters(self.opt_robot_model,self.opt_polytope_model)
        #self.opt_polytope_gradient_model.Gamma_hat_gradient(sigmoid_slope=1000)
        jac_output = Gamma_hat_gradient(J_Hess,Hess,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                        p_minus_hat,Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat,\
                        self.qdot_min,self.qdot_max,self.desired_vertices,sigmoid_slope=150)
        #print('jac_output',jac_output)

        #jac_output = sum(jac_output)
        return jac_output

    def hess_func(self,q_des):

        self.opt_robot_model.urdf_transform(q_joints=q_des)

        #self.opt_polytope_gradient_model.compute_polytope_gradient_parameters(self.opt_robot_model,self.opt_polytope_model)
        #self.opt_polytope_gradient_model.Gamma_hat_gradient(sigmoid_slope=1000)
        hess_output = self.opt_polytope_gradient_model.d_softmax_dq

        return hess_output









if __name__ == '__main__':
    	
    rospy.init_node('launchURrobot', anonymous=True)
    print("Started and Launched File \n")
    controller = LaunchRobot()
    rospy.spin()