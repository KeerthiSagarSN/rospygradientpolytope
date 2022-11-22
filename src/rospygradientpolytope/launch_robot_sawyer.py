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
from geometry_msgs import msg
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped,WrenchStamped, PointStamped
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Joy, JointState, PointCloud 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf_conversions as tf_c
from tf2_ros import TransformBroadcaster

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
from numpy import matrix,matmul,transpose,isclose,array,rad2deg,abs,vstack,hstack,shape,eye,zeros

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
base_link = "right_arm_base_link"

tip_link = "right_l6"




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

        self.cartesian_desired_vertices = 0.003*array([[0.20000, 0.50000, 0.50000],
								[0.50000, -0.10000, 0.50000],
								[0.50000, 0.50000, -0.60000],
								[0.50000, -0.10000, -0.60000],
								[-0.30000, 0.50000, 0.50000],
								[-0.30000, -0.10000, 0.50000],
								[-0.30000, 0.50000, -0.60000],
								[-0.30000, -0.10000, -0.60000]])
        
        
        self.desired_vertices = zeros(shape = (len(self.cartesian_desired_vertices),3))
        
        self.pub_rate = 500 #Hz

        self.sigmoid_slope = 150
        self.robot_joint_names = ['right_j0','right_j1','right_j2','right_j3','right_j4','right_j5','right_j6']

        self.robot_joint_names_pub = ['right_j0','head_pan','right_j1','right_j2','right_j3','right_j4','right_j5','right_j6']
        

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
        self.q_in = zeros(7)
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
        input('stop here')
        self.Gamma_min_softmax = None
        self.pos_desired = array([[0.45, 0.32, 0.525]])

        #self.q_joints_opt = None

        opt_solution = self.fmin_opt(self.pos_desired,analytical_solver = True)
        #print('self.pos_desired',self.pos_desired)
        print('opt_solution.x',opt_solution.x)
        self.pos_actual_mat = self.pykdl_util_kin.forward(opt_solution.x,tip_link,base_link)
        self.pos_actual = self.pos_actual_mat[0:3,3]

        print('self.pos_actual',self.pos_actual)
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
    def joint_state_callback(self,sawyer_joints):

        ## Get Joint angles of the Sawyer Robot
        self.q_in[0] = sawyer_joints.position[0]
        self.q_in[1] = sawyer_joints.position[2]
        self.q_in[2] = sawyer_joints.position[3]
        self.q_in[3] = sawyer_joints.position[4]
        self.q_in[4] = sawyer_joints.position[5]
        self.q_in[5] = sawyer_joints.position[6]
        self.q_in[6] = sawyer_joints.position[7]
        

        #self.q_in[3] = 0.5
        #self.q_in[4] = 0.89

        #print('self.q_in',self.q_in)
        #print('tip_link',tip_link)
        #print('base_link',base_link)
        ef = self.pykdl_util_kin.forward(self.q_in,tip_link,base_link)

        
        
    
    

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
        J_Hess = jacobianE0(q_in)
        #J_Hess_pykdl = array(self.pykdl_util_kin.jacobian(q_in))
        #J_Hess = J_Hess[0:3,:]
        ##print('J_Hess',J_Hess)
        #print('J_Hess_pykdl',J_Hess_pykdl)
        #input('wait here')
        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
            J_Hess,active_joints=7,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)


        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
            J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints=7,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)
        
        self.Gamma_min_softmax = Gamma_min_softmax
        #print('Current objective function in optimization is',-self.opt_polytope_model.Gamma_min_softmax)
        print('Current objective function in optimization is',-1.0*Gamma_min_softmax)
        

        #err_pos = abs((norm(pos_act-self.pos_reference) - 1e-6))
        #print('error',err_pos)
        #return err_pos
        return -1.0*self.Gamma_min_softmax


    def constraint_func(self,q_in):

        #pos_act_mat = self.pykdl_util_kin.forward(q_in,tip_link,base_link)
        
        #pos_act = pos_act_mat[0:3,3]

        pos_act = position_70(q_in)
        print('self.pos_reference',self.pos_reference)
        print('Current position in optimization is',pos_act)
        #input('Wait here ')
        return ( 1e-2 - norm(pos_act-self.pos_reference))



    ### Constraints should be actual IK - Actual vs desrired - Cartesian pos

    ## NOrm -- || || < 1eps-
    def constraint_func_Gamma(self):

        #self.opt_robot_model.urdf_transform(q_joints=q_des)
        #canvas_input.generate_axis()
        
        #J_Hess = array(self.pykdl_util_kin.jacobian(q_in))
        '''
        J_Hess = jacobianE0(q_in)
        #J_Hess = J_Hess[0:3,:]

        #print('self.qdot_min',-1*self.qdot_max)
        #print('self.qdot_max',self.qdot_max)
        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
            J_Hess,active_joints=7,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)


        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
            J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints=7,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)
        '''
        #print('Current objective in optimization Gamma is',self.opt_polytope_model.Gamma_min_softmax)
        return self.Gamma_min_softmax

    def jac_func(self,q_in):
        from numpy import sum

        #J_Hess = array(self.pykdl_util_kin.jacobian(q_in))
        J_Hess = jacobianE0(q_in)
        #J_Hess = J_Hess[0:3,:]
        #print('J_Hess',J_Hess)
        Hess = getHessian(J_Hess)

        #print('Hessian',Hess)

        #input('wait here for hessian')
        #J_Hess = J_Hess[0:3,:]

        #print('self.qdot_min',-1*self.qdot_max)
        #print('self.qdot_max',self.qdot_max)
        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(\
            J_Hess,active_joints=7,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
            qdot_max=self.qdot_max,cartesian_desired_vertices= self.desired_vertices,sigmoid_slope=150)


        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = get_capacity_margin(\
            J_Hess, n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints=7,cartesian_dof_input = array([True,True,True,False,False,False]),qdot_min=self.qdot_min,
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
    	
    rospy.init_node('launchSawyerrobot', anonymous=True)
    print("Started and Launched File \n")
    controller = LaunchRobot()
    rospy.spin()