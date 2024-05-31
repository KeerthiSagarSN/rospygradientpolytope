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

#!/usr/bin/env python3
## Library import
############# ROS Dependencies #####################################
import rospy
import os
from geometry_msgs import msg
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped, WrenchStamped, PointStamped
from std_msgs.msg import Bool, Float32,Int16,String

from sensor_msgs.msg import Joy, JointState, PointCloud
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial import distance as dist_scipy
from numpy import sum


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


from rospygradientpolytope.visual_polytope import velocity_polytope,desired_polytope,velocity_polytope_with_estimation
from rospygradientpolytope.polytope_ros_message import create_polytopes_msg, create_polygon_msg,create_capacity_vertex_msg, create_segment_msg
from rospygradientpolytope.polytope_functions import get_polytope_hyperplane, get_capacity_margin
from rospygradientpolytope.polytope_gradient_functions import Gamma_hat_gradient_dq,Gamma_hat_gradient, Gamma_hat_gradient_joint
from rospygradientpolytope.sawyer_functions import jacobianE0, position_70, jacobian70
from rospygradientpolytope.robot_functions import getHessian, exp_normalize

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



from re import T

################# URDF Parameter Server ##################################################
# URDF parsing an kinematics - Using pykdl_utils : For faster version maybe use PyKDL directly without wrapper
from urdf_parser_py.urdf import URDF

from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.joint_kinematics import JointKinematics



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

# Import URDF of the robot - # todo the param file for fetching the URDF without file location
## Launching the Robot here - Roslaunching an external launch file of the robot


# loading the root urdf from robot_description parameter
# Get the URDF from the parameter server
# Launch the robot 
robot_urdf = URDF.from_parameter_server('robot_description') 

### Base and tcp_link as described in the URDF file
base_link = "right_arm_base_link"

tip_link = "right_hand"




# Build kinematic chain here

class LaunchSawyerRobot():

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


        self.publish_vertex_pose = rospy.Publisher("/ef_pose_vertex", PointStamped, queue_size=1)
        self.publish_vertex_desired_pose = rospy.Publisher("/ef_desired_pose_vertex", PointStamped, queue_size=1)
        # Subscribe joints of Robot --- Joint State subscriber

        self.robot_joint_state_publisher = rospy.Publisher("/joint_states",JointState, queue_size = 1)

        #self.robot_joint_state_subscriber = rospy.Subscriber("/joint_states",JointState,self.joint_state_callback,queue_size=1)

        self.cartesian_desired_vertices = array([[0.20000, 0.50000, 0.50000],
                                        [0.50000, -0.10000, 0.50000],
                                        [0.50000, 0.50000, -0.60000],
                                        [0.50000, -0.10000, -0.60000],
                                        [-0.30000, 0.50000, 0.50000],
                                        [-0.30000, -0.10000, 0.50000],
                                        [-0.30000, 0.50000, -0.60000],
                                        [-0.30000, -0.10000, -0.60000]])
        
        
        self.desired_vertices = self.cartesian_desired_vertices
        self.sigmoid_slope_input = 150

        
        self.pub_rate = 500 #Hz

        #self.sigmoid_slope = 150
        # Manually declare to avoid ambiguity
        self.robot_joint_names = ['right_j0','right_j1','right_j2','right_j3','right_j4','right_j5','right_j6']

        self.robot_joint_names_pub = ['right_j0','head_pan','right_j1','right_j2','right_j3','right_j4','right_j5','right_j6']

        self.active_joints = len(self.robot_joint_names)
        

        # maximal joint velocities- From URDF file 
        self.qdot_limit = [robot_urdf.joint_map[i].limit.velocity for i in self.robot_joint_names]


        # maximal and minimal joint velocities
        self.qdot_max = array(self.qdot_limit)
        self.qdot_min = -1*self.qdot_max



        self.fun_iter = Int16()
        self.fun_iter.data = 0
        self.start_optimization_bool = False

        self.msg_status_ik = String()

        self.q_in = zeros(self.active_joints)


        jac_output = mp.Array("f",zeros(shape=(self.active_joints)))

        self.plot_polytope_thread = None
        self.thread_is_running = False


        ### All interactive IK attributes here

        self.msg_status_ik = String()
        # Create an interactive marker server        
        self.desired_pose = Pose()
        self.polytope_display = False
        
        self.polytope_display_on_sub = rospy.Subscriber("polytope_show",Bool,self.polytope_show_on_callback)
        self.start_interactive_ik_sub = rospy.Subscriber("run_ik",Bool,self.start_interactive_ik)
        self.pub_end_ik = rospy.Publisher("ik_progress",Int16,queue_size=1)
        self.pub_status_ik = rospy.Publisher("status_ik",String,queue_size=1)
        self.sub_ik_pos = rospy.Subscriber("interactive_sphere",Pose,self.ik_pose_callback)





        #self.q_in = array([0.3,0.5,0.2,0.3,0.1,0.65,0.9])
        #self.q_test = zeros(7)

        self.pykdl_util_kin = KDLKinematics(robot_urdf , base_link, tip_link,None)
        #self.q_bounds = zeros(len(self.q_upper_limit),2)
        
        self.q_upper_limit = array([self.pykdl_util_kin.joint_limits_upper]).T
        #self.q_upper_limit = self.pykdl_util_kin.joint_limits_upper
        self.q_lower_limit = array([self.pykdl_util_kin.joint_limits_lower]).T

        self.q_bounds = hstack((self.q_lower_limit,self.q_upper_limit))


        ########################################## Uncomment the below code to test numerical_vs_analytical gradient #####3
        #self.test_gamma_gradient(sigmoid_slope_test=sigmoid_slope_test)
        
        ##################################################################################################################















        ######################## Paper Results below- Do not uncomment for testing purposes #######################################################################
        '''
        BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/test_results/"
        test_case = 3
        self.test_case = test_case
        file_name = 'q_in_sawyer_test_'+str(test_case)
        '''
        ## Generate 
        '''
        
        ## First generate a list of random joints in numpy and save in numpy savez array - First generation
        self.q_in_array = zeros(shape = (1000,7))
        #for i in range(100):
        for j in range(7):
            self.q_in_array[:,j] = random.uniform(self.q_lower_limit[j,0],self.q_upper_limit[j,0],size=(1000))
        
        # Save the file as npz 
        #savez('q_in_sawyer', q_in_arr = self.q_in_array)
        savez(os.path.join(BASE_PATH, file_name),q_in_arr = self.q_in_array)
        '''


        
        # Load random joint configurations within the limit
        '''
        data_load = load(os.path.join(BASE_PATH, file_name)+str('.npz'))
        #q_in_array_load = load('q_in_sawyer_test_1.npz')
        self.q_in_array = data_load['q_in_arr']

        #file_name = 'ef_IK_Sawyer_'+str(test_case)

        
        test_case = 3
        self.test_case = test_case
        
        #file_name = 'ef_IK_UR_'+str(test_case)
        
        file_name = 'ef_IK_Sawyer_'+str(test_case)
        '''
        ## First generate a list of random end-effector points in numpy and save in numpy savez array - First generation
        '''
        self.ef_IK_array = zeros(shape = (100,3))

        for i in range(100):
            
            self.ef_IK_array[i,0] = random.uniform(0.5,1.0)
            self.ef_IK_array[i,1] = random.uniform(-0.5,0.5)
            self.ef_IK_array[i,2] = random.uniform(-0.3,1.0)
        
        # Save the file as npz 
        
        savez(os.path.join(BASE_PATH, file_name),ef_IK_arr = self.ef_IK_array)
        '''
        # Load random joint configurations within the limit
        '''
        data_load_IK = load(os.path.join(BASE_PATH, file_name)+str('.npz'))
        #q_in_array_load = load('q_in_sawyer_test_1.npz')
        self.ef_IK_array = data_load_IK['ef_IK_arr']

        
        ## Testing gamma vs gamma_hat here
        #self.pos_desired = array([[0.49, 0.412, 0.625]])
        sigmoid_slope_test = array([50,100,150,200,400])

        self.sigmoid_slope_array = array([50,100,150,200,400])

        #self.Error_gamma_array = zeros(shape=(len(self.q_in_array),len(self.sigmoid_slope_arra)))
        self.Error_IK_array = zeros(shape=(len(self.q_in_array),len(self.sigmoid_slope_array)))

        self.ts_arr = zeros(shape=(len(self.q_in_array),len(self.sigmoid_slope_array)))

        self.Gamma_min_array =  zeros(shape=(len(self.q_in_array),len(self.sigmoid_slope_array)))

        self.Gamma_min_softmax_array =  zeros(shape=(len(self.q_in_array),len(self.sigmoid_slope_array)))

        #self.test_Gamma_vs_Gamma_hat(self.sigmoid_slope_test)



        self.cm_est = None

        self.time_counter = 0
        self.fun_counter = 0

        self.color_array_cm = ['g','r']
        self.cm_est_arr = zeros(shape=(2))
        self.cm_est_arr[:] = -10000
        self.time_arr = zeros(shape=(2))

        self.test_gamma_gradient(sigmoid_slope_test=sigmoid_slope_test)
        self.fig_cm, self.ax_cm = plt.subplots()

        self.ax_cm.set_xlabel('Iterations',fontsize=13)
        self.ax_cm.set_ylabel('Estimated Capacity Margin'+  r"($\hat{\gamma}$)",fontsize=13)
        self.plt_obj = None
        plt.show()  

        

        input('Ctrl+C and exit ')
        #q_0 = zeros(6)
        #q_0 = float64(q_0)
        ##print('pos_des',float64(self.ef_IK_array[0,:]))
        #print('pos_des',float64(self.ef_IK_array[1,:]))
        #input('pos-desired')
        
        input('stop and exit dont test further')
        '''
        ### Save ik optimizaiton results
        '''
        for iii in range(50,100):
            for jjj in range(len(self.sigmoid_slope_array)):

                q_0 = self.q_in_array[iii,:]

                
                self.sigmoid_slope_input = self.sigmoid_slope_array[jjj]
                
                #pos_des = float64(array([0.0,1.00,1.00]))

                pos_des = float64(self.ef_IK_array[iii,:])

                print('pos_des',pos_des)
                print('q_0',q_0)
                print('delta_qmin',self.qdot_max)
                #input('desired pose')

                opt_result = self.fmin_opt(q_0, pos_des,True)

                BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/test_results/"
                file_name = 'ik_opt_gamma_obj_sawyer_steep_'+str(iii) + str('_') + str(self.sigmoid_slope_input) + str('_') + str(self.test_case)

                savez(os.path.join(BASE_PATH, file_name),opt_q =opt_result.x,opt_gamma =opt_result.fun,opt_success =opt_result.success,opt_message =opt_result.message,opt_iterations =opt_result.nit,\
                    sigmoid_slope = self.sigmoid_slope_input,q0_input  = q_0, ef_desired = pos_des)

        
        # Test success rate here
        
        
        # Number of success 
        #input('stop and exit dont test further')
        num_of_succ = zeros(shape=(100,len(self.sigmoid_slope_array)))
        obj_fun_analysis = zeros(shape=(100,len(self.sigmoid_slope_array)))
        # Number of iterations
        num_of_iteration = zeros(shape=(100,len(self.sigmoid_slope_array)))
        for iii in range(0,100):
            for jjj in range(len(self.sigmoid_slope_array)):

                q_0 = self.q_in_array[iii,:]

                
                self.sigmoid_slope_input = self.sigmoid_slope_array[jjj]
                
                #pos_des = float64(array([0.0,1.00,1.00]))

                pos_des = float64(self.ef_IK_array[iii,:])

                print('pos_des',pos_des)

                #input('desired pose')
                #opt_result = self.fmin_opt(q_0, pos_des,True)

                BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/test_results/"
                file_name = 'ik_opt_gamma_obj_sawyer_steep_'+str(iii) + str('_') + str(self.sigmoid_slope_input) + str('_') + str(self.test_case)

                # Load random joint configurations within the limit
                data_load = load(os.path.join(BASE_PATH, file_name)+str('.npz'))

                
                num_of_succ[iii,jjj] = data_load['opt_success']
                if num_of_succ[iii,jjj]:
                    num_of_iteration[iii,jjj] = data_load['opt_iterations']
                    obj_fun_analysis[iii,jjj] = data_load['opt_gamma']

            
            #print('Function analysis',obj_fun_analysis[iii,:])
            #input('stop and test')
        
        print('Number of success - 50',count_nonzero(num_of_succ[:,0]))
        print('Number of success - 100',count_nonzero(num_of_succ[:,1]))
        print('Number of success - 150',count_nonzero(num_of_succ[:,2]))
        print('Number of success - 200',count_nonzero(num_of_succ[:,3]))
        print('Number of success - 400',count_nonzero(num_of_succ[:,4]))





        print('Number of iters - 50',sum(num_of_iteration[:,0])/count_nonzero(num_of_iteration[:,0]))
        print('Number of iters - 100',sum(num_of_iteration[:,1])/count_nonzero(num_of_iteration[:,1]))
        print('Number of iters - 150',sum(num_of_iteration[:,2])/count_nonzero(num_of_iteration[:,2]))
        print('Number of iters - 200',sum(num_of_iteration[:,3])/count_nonzero(num_of_iteration[:,3]))
        print('Number of iters - 400',sum(num_of_iteration[:,4])/count_nonzero(num_of_iteration[:,4]))


        print('Average of gamma - 50',sum(obj_fun_analysis[:,0])/count_nonzero(num_of_iteration[:,0]))
        print('Average of gamma - 100',sum(obj_fun_analysis[:,1])/count_nonzero(num_of_iteration[:,1]))
        print('Average of gamma - 150',sum(obj_fun_analysis[:,2])/count_nonzero(num_of_iteration[:,2]))
        print('Average of gamma - 200',sum(obj_fun_analysis[:,3])/count_nonzero(num_of_iteration[:,3]))
        print('Average of gamma - 400',sum(obj_fun_analysis[:,4])/count_nonzero(num_of_iteration[:,4]))
        '''
        # print nothing

        '''

        input('started checking gradient')
        gradient_result = check_grad(self.obj_function,self.jac_func,x0 = self.q_in,epsilon=1.0e-09)
        print('gradient_result',gradient_result )
        input('stopped checking gradient')


        #self.q_joints_opt = None

        opt_solution = self.fmin_opt(self.pos_desired,analytical_solver = True)
        #print('self.pos_desired',self.pos_desired)
        print('opt_solution.x',opt_solution.x)
        self.pos_actual_mat = self.pykdl_util_kin.forward(opt_solution.x,tip_link,base_link)
        self.pos_actual = self.pos_actual_mat[0:3,3]

        print('self.pos_actual',self.pos_actual)

        ### Check gradient here

        input('stop the gradient check here')
        '''
        ## Stop here too

        


        '''
        while not rospy.is_shutdown():
            mutex.acquire()
            msg = JointState()

            msg.name = [self.robot_joint_names[0], self.robot_joint_names[1],self.robot_joint_names[2],self.robot_joint_names[3],
                        self.robot_joint_names[4], self.robot_joint_names[5], self.robot_joint_names[6], self.robot_joint_names[7]]
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'sawyer_base'
            msg.position = [0,0.0,0.0,0.05,0.0,0.05,0.0,0.05]
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
    
        ######################## Paper Results above - DO not uncomment above for testing #######################################################################


    def ik_pose_callback(self,desired_ik_pose):
        self.desired_ik_pose = desired_ik_pose.position
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




        
        #print('self.polytope_display',self.polytope_display)

    def plot_polytope(self):
        
        while self.thread_is_running:
            if self.polytope_display:

                
                #print('polytope plotting')


                mutex.acquire()
                pos_act = array(self.pykdl_util_kin.forward(self.q_in)[0:3, 3])

                #pos_act = array(self.pykdl_util_kin._do_kdl_fk(q_in,5)[0:3,3])

                J_Hess = array(self.pykdl_util_kin.jacobian(self.q_in))

                # print('pos_act_mat',pos_act_mat)

                #pos_act = position_70(q_in)
                
                

                scaling_factor = 10.0
                
                ### Polytope plot with estimation
                
                #pykdl_kin_jac = pykdl_util_kin.jacobian(self.q_in_numpy)
                polytope_verts, polytope_faces, facet_vertex_idx, capacity_faces, capacity_margin_proj_vertex, \
                    polytope_verts_est, polytope_faces_est, capacity_faces_est, capacity_margin_proj_vertex_est,cm_index = \
                                                velocity_polytope_with_estimation(J_Hess,self.qdot_max,self.qdot_min,self.desired_vertices,self.sigmoid_slope_input)
                desired_polytope_verts, desired_polytope_faces = desired_polytope(self.desired_vertices)


                self.cm_est = cm_index

                

                # Only for visualization - Polytope at end-effector - No physical significance
                #ef_pose = pykdl_util_kin.forward(self.q_in_numpy)[:3,3]

                
                

                # Get end-effector of the robot here for the polytope offset

                #ef_pose = position_70(q_in)

                ef_pose = pos_act
                ef_pose = ef_pose[:,0]

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


                ### Vertex for capacity margin on the Available Polytope
                ActualposevertexArray_message = self.publish_vertex_pose.publish(create_capacity_vertex_msg(ef_pose, \
                                                                                            array([0,0,0]), "right_arm_base_link", 1))
                
                ### Vertex for capacity margin on the Available Polytope
                '''
                DesiredposevertexArray_message = self.publish_vertex_desired_pose.publish(create_capacity_vertex_msg(self.pos_reference, \
                                                                                        array([0,0,0]), "base_link", 1))
                '''
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
                
                
                #print('facet_vertex_idx',facet_vertex_idx)
                
                mutex.release()


    ### Start Interactive IK computation ######################
    def compute_pose_ik(self, pos_ik):
        import time
        from numpy.linalg import det
        from numpy import sum, mean, average, linspace
        import matplotlib.pyplot as plt

        
        q0 = zeros(shape=(self.active_joints))
        for j in range(self.active_joints):
            q0[j] = random.uniform(
                self.q_lower_limit[j, 0], self.q_upper_limit[j, 0])
        

        #for i in range(0, 1000):

        st = time.time()
        q_opt = self.fmin_opt(q0, pos_ik, True)
        # To publish the joint states to Robot
        self.joint_state_publisher_robot(q_opt.x)

        ex_time = time.time() - st
        print('execution time is',ex_time)
        q0 = q_opt.x


    ### Numerical testing - Error between classical capacity margin and 
    #          estimated capacity margin        ######################

    def test_Gamma_vs_Gamma_hat(self,sigmoid_slope_test):
        
        import time
        from numpy.linalg import det
        from numpy import sum,mean,average
        import matplotlib.pyplot as plt

        
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
        #q_add = zeros(7)
        #q_add[test_joint] = 1
        step_size = 0.001



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



                # Publish joint sates to Rviz for the robot  
                # #self.joint_state_publisher_robot()            


                J_Hess = array(self.pykdl_util_kin.jacobian(q_in,pos=None))


                h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
                    J_Hess,active_joints=self.active_joints,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
                    qdot_max=self.qdot_max,cartesian_desired_vertices= self.cartesian_desired_vertices,sigmoid_slope=sigmoid_slope_inp )


                Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
                    J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                                active_joints=self.active_joints,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
                    qdot_max=self.qdot_max,cartesian_desired_vertices= self.cartesian_desired_vertices,sigmoid_slope=sigmoid_slope_inp )
            
                
                
                self.Error_gamma_array[i,j] = (Gamma_min - Gamma_min_softmax)#/(1.0*Gamma_min))*100 
                

                self.Gamma_min_array[i,j] = Gamma_min

                self.Gamma_min_softmax_array[i,j] = Gamma_min_softmax

                self.ts_arr[i,j] = (time.time() - start_time)              


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
        # List of labels from sigmoid slope
        ax.set_xticklabels(['','50','100','150','200','400'])
        plt.xlabel("Sigmoid Slope",size=15)
        plt.ylabel("Error",size=15)
        # Create the boxplot
        bp = ax.violinplot(data_to_plot)
        for pc in bp['bodies']:
            pc.set_facecolor('orange')
            pc.set_edgecolor('orange')

        #ax.set_aspect(1)
        
        plt.savefig('Error_plot_sawyer_' + str(self.test_case)+('.png'))
        plt.show()


        file_name = 'error_plot_sawyer_'+str(self.test_case)

        savez(os.path.join(BASE_PATH, file_name),q_in = self.q_in_array,sigmoid_slope_inp = sigmoid_slope_test, ts = self.ts_arr,\
            Gamma_min_array = self.Gamma_min_array, Gamma_min_softmax_array = self.Gamma_min_softmax_array,Error_gamma_array = self.Error_gamma_array)

        fig2, ax2 = plt.subplots()
        # List of five airlines to plot
        sigmoid_slope_plot = ['50', '100', '150', '200', '400']

        
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

        

        plt.show()

    #### Testing numerical gradient of classical capacity margin vs analytical gradient of estimated capacity margin
    def test_gamma_gradient(self,sigmoid_slope_test):
        import time
        from numpy.linalg import det
        from numpy import sum,mean,average
        import matplotlib.pyplot as plt




        Gamma_min_prev = None

        # test_joint = 1
        # q_in = zeros(7)
        # #q_add[test_joint] = 1
        
        # q_in[0] = -1.57
        # q_in[1] = -0.97
        # q_in[2] = -0.94
        # q_in[3] = 0.32
        # q_in[4] = 0.15
        # q_in[5] = -0.12
        # q_in[6] = -0.76
        # step_size = 0.0020

        num_iterations = 500
        #sigmoid_slope_inp = 150

        i0_plot = zeros(shape=(num_iterations))
        #len(x0_start)
        sigmoid_slope_arr = [50,100.0,150.0,200.0,400.0]
        error_plot_a = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
        error_plot_n = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
        z0_plot = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
        x0_plot = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
        y0_plot = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))

        color_arr = ['magenta','k','green','cyan','blue','r']
        for lm in range(0,1):
        #for lm in range(len(sigmoid_slope_arr)):
            sigmoid_slope_inp = sigmoid_slope_arr[lm]

            test_joint = 3
            q_in = zeros(self.active_joints)
            #q_add[test_joint] = 1
            
            ## Paper values
            q_in[0] = 0.0
            q_in[1] = 0.0
            q_in[2] = -1.625
            q_in[3] = 0.0
            q_in[4] = 0.10
            q_in[5] = -0.12
            q_in[6] = -0.05
            

            ### Declare appropriate step size for the numerical gradient
            ## Smaller the step size- much smoother
            step_size = 0.001

            
            ax2 = plt.axes()
            ax2.set_xlabel('Joint q' + str(test_joint)+' (rad)')
            ax2.set_ylabel('Capacity Margin Gradient' + str(' [N]'),fontsize=13)
            ax2.plot([],[],color='r',linestyle='solid',label='Numerical Gradient: ' + r"$\frac{\partial {\gamma}}{\partial{q_3}}$")
            ax2.plot([],[],color='g',linestyle='solid',label='Analytical Gradient- Slope 50')
            ax2.legend()
            
            '''
            ax2.legend()
            plt.show()
            '''
            #plt.legend(loc="lower right")
            numerical_gradient_arr = zeros(shape=(num_iterations))
            analytical_gradient_arr = zeros(shape=(num_iterations))
            x_arr = zeros(shape=(num_iterations))
            for i in range(num_iterations):
                
                i0_plot[i] = i

                self.q_in = q_in

                q_in[test_joint] += step_size
                x_arr[i] = q_in[test_joint]

                i0_plot[i] = q_in[test_joint]


                self.joint_state_publisher_robot(q_in)
                #time.sleep(0.005)

                # Publish joint sates to Rviz for the robot
                

                J_Hess = array(self.pykdl_util_kin.jacobian(q_in))

                Hess = getHessian(J_Hess)

                h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
                    J_Hess,active_joints=self.active_joints,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
                    qdot_max=self.qdot_max,cartesian_desired_vertices= self.cartesian_desired_vertices,sigmoid_slope=sigmoid_slope_inp )


                Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
                    J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                                active_joints=self.active_joints,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
                    qdot_max=self.qdot_max,cartesian_desired_vertices= self.cartesian_desired_vertices,sigmoid_slope=sigmoid_slope_inp )
                
                
                jac_output = Gamma_hat_gradient(J_Hess,Hess,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                    p_minus_hat,Gamma_total_hat, Gamma_min_index_hat,\
                    self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,sigmoid_slope=sigmoid_slope_inp)


                if Gamma_min_prev != None:

                    #for i in range(0,limit,step_size):
                    numerical_gradient = (Gamma_min_prev - Gamma_min) /(step_size*1.0)
                    error_plot_n[i,lm] = numerical_gradient
                    numerical_gradient_arr[i] = numerical_gradient

                    print('numerical_gradient',numerical_gradient)

                    analytical_gradient = jac_output[test_joint]

                    error_plot_a[i,lm] = analytical_gradient
                    analytical_gradient_arr[i] = analytical_gradient
                    print('analytical_gradient',analytical_gradient)
                    #plt.cla()
                    if i > 1:
                        #ax2.scatter(q_in[test_joint],numerical_gradient,color='r',s=2.5)
                        #ax2.scatter(q_in[test_joint],analytical_gradient,color='k',s=2.5)
                        x1 = [x_arr[i-1],x_arr[i]] 
                        y1 = [numerical_gradient_arr[i-1],numerical_gradient_arr[i]]
                        y2 = [analytical_gradient_arr[i-1],analytical_gradient_arr[i]]
                        ax2.plot(x1,y1,color='r')
                        ax2.plot(x1,y2,color='g')
                        
                        plt.pause(0.00001)
                    

                    #error_grad = ((numerical_gradient - analytical_gradient)) /(numerical_gradient*1.0)*100.0
                    #error_grad[i,lm] = error_grad

                Gamma_min_prev = Gamma_min
            

                
                ############### Plotting the polytope while doing gradient computation ########################
                ########### 
                
            
                scaling_factor = 10.0
                #mutex.acquire()
                ### Polytope plot with estimation
                
                #pykdl_kin_jac = pykdl_util_kin.jacobian(self.q_in_numpy)
                polytope_verts, polytope_faces, facet_vertex_idx, capacity_faces, capacity_margin_proj_vertex, \
                    polytope_verts_est, polytope_faces_est, capacity_faces_est, capacity_margin_proj_vertex_est, cm_est = \
                                                velocity_polytope_with_estimation(J_Hess,self.qdot_max,self.qdot_min,self.cartesian_desired_vertices,sigmoid_slope_inp)
                desired_polytope_verts, desired_polytope_faces = desired_polytope(self.cartesian_desired_vertices)



                

                # Only for visualization - Polytope at end-effector - No physical significance
                #ef_pose = pykdl_util_kin.forward(self.q_in_numpy)[:3,3]

                
                

                # Get end-effector of the robot here for the polytope offset

                #ef_pose = position_70(q_in)

                ef_pose = self.pykdl_util_kin.forward(self.q_in)[:3,3]

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
                
                ######### Comment above section for faster gradient computation #############################################
                ##############################################################################################################
                
                
                #print('facet_vertex_idx',facet_vertex_idx)
            
            #mutex.release()
        

        ### Plot the result of the gradient comparison below ####################################################################
        ax2 = plt.axes()
        ax2.set_xlabel('Joint q' + str(test_joint),fontsize=13)
        #ax2.set_ylabel(r"$\partial \hat{\gamma}$" + str(' [N]'),fontsize=13)
        ax2.set_ylabel('Capacity Margin Gradient' + str(' [N]'),fontsize=13)


        #handle_1 = ax2.scatter(i0_plot[1],error_plot_a[1,1],color='k',s=0.0000001,label='Estimated:'+ r"$\frac{\partial \hat{\gamma}}{\partial{q_3}}$")

        handle_2 = ax2.plot(i0_plot[1:],error_plot_a[1:,0],color=color_arr[0],linestyle='dashed',label='Analytical Slope: 50')
        handle_3 = ax2.plot(i0_plot[1:],error_plot_a[1:,1],color=color_arr[1],linestyle='dashed',label='Analytical Slope: 100')
        handle_4 = ax2.plot(i0_plot[1:],error_plot_a[1:,2],color=color_arr[2],linestyle='dashed',label='Analytical Slope: 150')
        handle_5 = ax2.plot(i0_plot[1:],error_plot_a[1:,3],color=color_arr[3],linestyle='dashed',label='Analytical Slope: 200')
        handle_6 = ax2.plot(i0_plot[1:],error_plot_a[1:,4],color=color_arr[4],linestyle='dashed',label='Analytical Slope: 400')


        #handle_6 = ax2.scatter(i0_plot[1],error_plot_a[1,1],color='k',s=0.0000001,label='Actual' + r"$\frac{\partial {\gamma}}{\partial{q_3}}$")
        #input('second plot')
        label_numerical = ax2.plot(i0_plot[1:],error_plot_n[1:,0],color=color_arr[5],linestyle='solid',label='Numerical Gradient: ' + r"$\frac{\partial {\gamma}}{\partial{q_3}}$")
        #ax2.plot(i0_plot[1:],error_plot_n[1:,1],color=color_arr[1],linestyle='solid',label='Numerical Slope: 150')
        #ax2.plot(i0_plot[1:],error_plot_n[1:,2],color=color_arr[2],linestyle='solid',label='Numerical Slope: 200')
        #ax2.plot(i0_plot[1:],error_plot_n[1:,3],color=color_arr[3],linestyle='solid',label='Numerical Slope: 400')
        # reordering the labels

        
        # pass handle & labels lists along with order as below
        #handle_str_1 = 'Estimated:' + r"$\frac{\partial \hat{\gamma}}{\partial{q_3}}$"
        #handle_str_2 = 'Actual' + r"$\frac{\partial {\gamma}}{\partial{q_3}}$"
        #plt.legend((handle_1,handle_2,handle_3,handle_4,handle_5,handle_6,label_numerical),(handle_str_1,'Analytical Slope: 100','Analytical Slope: 150','Analytical Slope: 200','Analytical Slope: 400',handle_str_2,'Numerical Gradient'),loc="upper left")
        
        plt.legend(loc="lower right",fontsize='large')
        #plt.savefig('Sawyer_gradient_comparison_' + str(101)+('.png'))
        plt.savefig("Sawyer_gradient_comparison_111",format='png', dpi=600)

        plt.show()
        ####################### End plot #############################################################################
        ##############################################################################################################
    

    def joint_state_publisher_robot(self,q_joints):
        

        
        q_in = q_joints
        msg = JointState()

        msg.name = [self.robot_joint_names_pub[0], self.robot_joint_names_pub[1],self.robot_joint_names_pub[2],self.robot_joint_names_pub[3],
                    self.robot_joint_names_pub[4], self.robot_joint_names_pub[5], self.robot_joint_names_pub[6], self.robot_joint_names_pub[7]]
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'sawyer_base'
        msg.position = [q_in[0],0.0,q_in[1],q_in[2],q_in[3],q_in[4],q_in[5],q_in[6]]
        msg.velocity = []
        msg.effort = []        
        self.robot_joint_state_publisher.publish(msg)

    def joint_state_callback(self,sawyer_joints):

        ## Get Joint angles of the Sawyer Robot
        self.q_in[0] = sawyer_joints.position[0]
        ## Don't panic - Doing this below as the head- sawyer_joints.position[1] is not required for the chain
        self.q_in[1] = sawyer_joints.position[2] 
        self.q_in[2] = sawyer_joints.position[3]
        self.q_in[3] = sawyer_joints.position[4]
        self.q_in[4] = sawyer_joints.position[5]
        self.q_in[5] = sawyer_joints.position[6]
        self.q_in[6] = sawyer_joints.position[7]
        

        #### Computation of forward kinematics here using PyKDL 
        ef_pose = self.pykdl_util_kin.forward(self.q_in,tip_link,base_link)[0:3, 3] 

    

    ########### Main Optimizer Function ################################
    def fmin_opt(self, x0_start, pose_desired, analytical_solver: bool):
        #### Input ########
        # Initial point - x0_start
        # IK- pose required - pose_desired
        # Solver type: analytical_solver: True - SLSQP
        # Solver type: analytical_solver: False - COBYLA (Not extensively tested)
        
        #### Output ########

        # Joint positions for the desired IK position - q_joints_opt 

        ####################
        
        ## Converting datatype to float64 under assumption optimizer is fast with this
        ## TODO Need to compare if it is faster or not with float64

        self.initial_x0 = float64(x0_start)
        self.pos_reference = float64(pose_desired)

        # Printing for vanity check
        print('Reference position is', self.pos_reference)

        

        # Bounds created from the robot angles - Joint angle limits from URDF file
        ## So optimizer tries to give only feasible joint positions
        self.opt_bounds = float64(self.q_bounds)

        '''
        Paper value here
        cons = ({'type': 'ineq', 'fun': self.constraint_func},\
                {'type': 'ineq', 'fun': self.constraint_func_Gamma})
        '''
        #cons = ({'type': 'ineq', 'fun': self.constraint_function,'jac': self.jac_func})

        ## Video constraints here- Equality hard constraint leads to much faster convergence
        cons = ({'type': 'eq', 'fun': self.constraint_function, 'tol': 1e-5} )
               
            

        if analytical_solver:
            q_joints_opt = sco.minimize(fun=self.obj_function_gamma,  x0=self.initial_x0, bounds=self.opt_bounds,
                                        jac=self.jac_func, constraints=cons, method='SLSQP',
                                        options={'disp': True, 'maxiter': 200})  # Paper maximum iterations is 3000
        else:
            
            # Not tested extensively- May not function
            # Doesnt require an explicit jacobian
            q_joints_opt = sco.minimize(fun=self.obj_function_gamma,  x0=self.initial_x0, bounds=self.opt_bounds,
                                        constraints=cons, tol=1e-5, method='COBYLA',
                                        options={'disp': True})

        
        ### number of iterations before the optimization terminates
        ### TODO - In the interactive IK give user control to input the number of iterations
        ### This means that the optimizer is completed- Report 100% completion in the interactive IK panel
       
        self.fun_iter.data = 100

        ## This publishes to the status bar in the Interactive IK panel
        self.pub_end_ik.publish(self.fun_iter)
        if q_joints_opt.success:
            
            # Report the status of the optimizer result to the interactive IK panel
            # Successful convergence to the optimal capacity margin with the tolerance to the end-effector position
            # Directly from sco.minimze we fetch the result
            self.msg_status_ik.data = 'Success'
            
        else:
            # If the optimizer is still searching for the optimal and number of function evaluations > Iteration limit
            self.msg_status_ik.data = 'Time limit'            
        
        ## This publishes to the status bar in the Interactive IK panel

        self.pub_status_ik.publish(self.msg_status_ik)

        ## printing for vanity
        print('q_joints_opt', q_joints_opt.x)

        ## Stop updating the polytope display in the interactive IK panel
        self.polytope_display = False

        ## Return the final joint states for the required desired pose
        return q_joints_opt
        

    def obj_function_gamma(self, q_in):

        from numpy.linalg import det
        from numpy import sum

       

        # To publish the joint states to Robot
        self.joint_state_publisher_robot(q_in)
        self.q_in = q_in
        
        ## Compute the Jacobian for the current joint configuration
        J_Hess = array(self.pykdl_util_kin.jacobian(q_in))

        # Compute all hyperplane parameters
        h_plus, h_plus_hat, h_minus, h_minus_hat, p_plus, p_minus, p_plus_hat, p_minus_hat, n_k, Nmatrix, Nnot = get_polytope_hyperplane(
            J_Hess, active_joints=self.active_joints, cartesian_dof_input=array([True, True, True, False, False, False]), qdot_min=self.qdot_min,
            qdot_max=self.qdot_max, cartesian_desired_vertices=self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input)

        # Compute all estimated capacity margin parameters
        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(
            J_Hess, n_k, h_plus, h_plus_hat, h_minus, h_minus_hat,
            active_joints=self.active_joints, cartesian_dof_input=array([True, True, True, False, False, False]), qdot_min=self.qdot_min,
            qdot_max=self.qdot_max, cartesian_desired_vertices=self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input)

        # Computed smooth estimated capacity margin for the current joint configuration
        self.Gamma_min_softmax = Gamma_min_softmax
        

        # Negative sign manipulation- due to minimization in optimization
        return -1.0*self.Gamma_min_softmax


    ## Objective function- is the minimization of the error norm between desired 
    ## end-effector position and required end-effector position
    def obj_function_IK(self,q_in):
        
        # To plot the robot configuration
        #self.joint_state_publisher_robot(q_in)
        pos_act = array(self.pykdl_util_kin.forward(q_in)[0:3,3])

        return_dist_error = norm(pos_act.flatten()-self.pos_reference)

        # Float64- Assumption that it makes convergence faster

        return  float64(return_dist_error)
    
    def constraint_function(self, q_in):
        

        ### Input #######
        ## Current joint configuration
        # q_in

        ### Output #######
        ## euclidean distance norm - between current and desired end-effector position
        # norm(self.pos_reference-pos_act.flatten())

        # To publish the robot configuration- uncomment below
        # self.joint_state_publisher_robot(q_in)
        # Compute Forward kinematics - End-effector pose in the current q_in configurations
        pos_act = array(self.pykdl_util_kin.forward(q_in)[0:3, 3])        
        ef_pose = pos_act
        ef_pose = ef_pose[:, 0]        


        ### Vertex for capacity margin on the Available Polytope
        ActualposevertexArray_message = self.publish_vertex_pose.publish(create_capacity_vertex_msg(ef_pose, \
                                                                                            array([0,0,0]), "right_arm_base_link", 1))
                

        

        return -float64(norm(self.pos_reference-pos_act.flatten()))



    ### Constraints should be actual IK - Actual vs desrired - Cartesian pos

    ## NOrm -- || || < 1eps-
    def constraint_function_Gamma(self,q_in):


        
        J_Hess = array(self.pykdl_util_kin.jacobian(q_in))

        ### Compute- hyperplane parameters
        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
            J_Hess,active_joints=7,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=self.sigmoid_slope_input)

        ### Compute- estimated capacity margin parameters below
        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
            J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints=self.active_joints,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=self.sigmoid_slope_input)
        
        ## Estimed capacity margin
        return float64(1.0*Gamma_min_softmax)

    def jac_func(self, q_in):

        ### Input #######
        ## Current joint configuration
        # q_in

        ### Output #######
        ## analytical gradient of the estimated capacity margin with respect to the input joint configuration
        

        self.fun_counter += 0.5
        self.fun_iter.data = int(self.fun_counter)
        
        J_Hess = array(self.pykdl_util_kin.jacobian(q_in))

        Hess = getHessian(J_Hess)
        jac_output = mp.Array('f',zeros(shape=(self.active_joints)))


        h_plus, h_plus_hat, h_minus, h_minus_hat, p_plus, p_minus, p_plus_hat, p_minus_hat, n_k, Nmatrix, Nnot = get_polytope_hyperplane(
            J_Hess, active_joints=self.active_joints, cartesian_dof_input=array([True, True, True, False, False, False]), qdot_min=self.qdot_min,
            qdot_max=self.qdot_max, cartesian_desired_vertices=self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input)

        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(
            J_Hess, n_k, h_plus, h_plus_hat, h_minus, h_minus_hat,
            active_joints=self.active_joints, cartesian_dof_input=array([True, True, True, False, False, False]), qdot_min=self.qdot_min,
            qdot_max=self.qdot_max, cartesian_desired_vertices=self.desired_vertices, sigmoid_slope=self.sigmoid_slope_input)



        # Create a new thread and start it
        ## Multithreading- Faster gradient evaluation here
        threads = []
        for i_thread in range(self.active_joints):
            thread = mp.Process(target=Gamma_hat_gradient_dq,args=(J_Hess, Hess, n_k, Nmatrix, Nnot, h_plus_hat, h_minus_hat, p_plus_hat,\
                                        p_minus_hat, Gamma_total_hat, Gamma_min_index_hat,\
                                        self.qdot_min, self.qdot_max, self.desired_vertices,self.sigmoid_slope_input,i_thread,jac_output))
            thread.start()
            threads.append(thread)
        
        # now wait for them all to finish
        for thread in threads:
            thread.join()

        
        self.pub_end_ik.publish(self.fun_iter)
        
        return float64(jac_output)





if __name__ == '__main__':
    	
    rospy.init_node('launchSawyerrobot', anonymous=True)
    print("Started and Launched File \n")
    controller = LaunchSawyerRobot()
    rospy.spin()