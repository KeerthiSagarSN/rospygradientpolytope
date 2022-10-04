# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:44:09 2022

@author: keerthi.sagar
"""
import scipy.optimize as sco
from numpy import array,pi
from numpy.linalg import norm
from numpy.random import randn
from GenerateCanvas import GenerateCanvas
import cProfile

class OptimizationModel:

    def __init__(self):


        self.q_joints_input = None
        self.q_joints_opt = None
        #self.q_joints = None
        #self.obj_function = None
        self.function_opt = None

        self.gamma_input = None
        self.func_deriv = None
        self.bnds = (-1.0,1.0)

        self.initial_x0 = None
        self.cons = None
        self.opt_robot_model = None
        self.opt_polytope_model = None
        self.opt_polytope_gradient_model = None
        self.qdot_min = None
        self.qdot_max = None
        self.desired_vertices = None

        ##### Optimization PLot

        self.canvas_input_opt = None

        '''
        self.q1_bnds = [-pi,pi]
        self.q2_bnds = [-pi,pi]
        self.q3_bnds = [-pi,pi]
        self.q4_bnds = [-pi,pi]
        self.q5_bnds = [-pi,pi]
        self.q6_bnds = [-pi,pi]
        '''

        self.opt_bounds = None

        '''
        self.q1_act = 0.0
        self.q2_act = 0.0
        self.q3_act = 0.0
        self.q4_act = 0.0
        self.q5_act = 0.0
        self.q6_act = 0.0



        self.q1_des = 0.0
        self.q2_des = 0.0
        self.q3_des = 0.0
        self.q4_des = 0.0
        self.q5_des = 0.0
        self.q6_des = 0.0
        '''
        self.pos_reference = None
    ## Constraint equations - Cons1

    #ef constraint1(self):



    ## Constraint equations - Cons2



    def fmin_opt(self,robot_model, polytope_model, polytope_gradient_model,desired_vertices_inp,reference_pos,sigmoid_slope_inp,analytical_solver:bool):
        ### Function - func
        ## Initial point - x0
        ## args -
        ## method - SLQSQ
        ## jac = Jacobian - gradient of the

        ###
        #self.canvas_input_opt = GenerateCanvas()
        #self.canvas_input_opt.set_params(view_ang1=30,view_ang2=45,x_limits=[-20,20],y_limits=[-20,20],z_limits=[-20,20],axis_off_on = True)


        self.sigmoid_slope = sigmoid_slope_inp


        self.desired_vertices = desired_vertices_inp
        self.opt_robot_model = robot_model
        self.opt_polytope_model = polytope_model
        self.opt_polytope_gradient_model = polytope_gradient_model

        self.q_joints_input = robot_model.q_joints


        self.qdot_min = polytope_model.qdot_min
        self.qdot_max = polytope_model.qdot_max

        #self.obj_function = polytope_model.Gamma_total
        #print('self.obj_function',self.obj_function)

        self.initial_x0 = robot_model.q_joints


        #self.initial_x0 = randn(6)
        #print('self.initial_x0 ',self.initial_x0 )
        #print('self.obj_function(robot_model.q_joints)',self.obj_function(robot_model.q_joints))
        #self.func_deriv = polytope_gradient_model.d_gamma_hat


        self.opt_polytope_gradient_model.compute_polytope_gradient_parameters(self.opt_robot_model,self.opt_polytope_model)
        self.opt_polytope_gradient_model.Gamma_hat_gradient(sigmoid_slope=self.sigmoid_slope)
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

        self.pos_reference = reference_pos
        #print('Reference position is',self.pos_reference)


        ### Desired vertex set

        self.desired_vertices[:,0] = self.desired_vertices[:,0] + self.pos_reference[0]
        self.desired_vertices[:,1] = self.desired_vertices[:,1] + self.pos_reference[1]
        self.desired_vertices[:,2] = self.desired_vertices[:,2] + self.pos_reference[2]
        #print('self.opt_polytope_gradient_model.d_gamma_hat',self.opt_polytope_gradient_model.d_gamma_hat)

        # Bounds created from the robot angles

        self.opt_bounds = self.opt_robot_model.q_joints_bounds

        #print('self.opt_bounds is',self.opt_bounds)

        ### Constraints
        cons = ({'type': 'ineq', 'fun': self.constraint_func},\
                {'type': 'ineq', 'fun': self.constraint_func_Gamma})

        '''
        cons = ({'type': 'ineq', 'fun': lambda x:  self.q_joints_input[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
        '''
        ## jACOBIAN MATRIX OF THE OBJECTIVE FUNCTION

        # jac = self.opt_polytope_gradient_model.d_gamma_hat,

        ### WIth analyitacl gradient

        if analytical_solver:

            self.q_joints_opt = sco.minimize(fun = self.obj_function,  x0 = self.initial_x0,bounds = self.opt_bounds,\
                                         jac =self.jac_func, constraints = cons,method='SLSQP', \
                                             options={'disp': True,'maxiter':500})
        else:

            self.q_joints_opt = sco.minimize(fun = self.obj_function,  x0 = self.initial_x0,bounds = self.opt_bounds,\
                                             jac = None, constraints = cons,method='COBYLA', \
                                                 options={'disp': True,'maxiter':250})







        #hess = self.hess_func,
        #sco.check_grad(func = self.obj_function, grad =self.opt_polytope_gradient_model.d_gamma_hat \                                  , x0 = self.initial_x0, epsilon=1.4901161193847656e-08, direction='all', seed=None)
        #def constraint2(self):
    ## Obj
    def obj_function(self,q_des):



        from numpy.linalg import det
        from numpy import sum

        #input('inside obj func')
        #self.canvas_input_opt.generate_axis()
        #self.opt_robot_model.urdf_transform(q_joints=q_des)
        #canvas_input.generate_axis()



        self.opt_polytope_model.generate_polytope_hyperplane(self.opt_robot_model,cartesian_dof_input = array([True,True,True,False,False,False]),
                                            qdot_min=self.qdot_min, qdot_max=self.qdot_max, \
                                            cartesian_desired_vertices = self.desired_vertices,sigmoid_slope = self.sigmoid_slope)
        #self.opt_polytope_model.plot_polytope(self.canvas_input_opt,True,False)
        '''
        if self.opt_polytope_model.Gamma_min_softmax < 0:
            print('self.opt_polytope_model.Gamma_min_softmax',self.opt_polytope_model.Gamma_min_softmax )
            print('infeasible joint angles are:',q_des)
            print('jacobian feasilbilty',det(self.opt_robot_model.jacobian_hessian))
            print('jacobian is',self.opt_robot_model.jacobian_hessian)
            #print('joint angle bounds are',self.opt_bounds)
            #input('Infeasible')
        '''


        print('Current objective function in optimization is',-self.opt_polytope_model.Gamma_min_softmax)

        return -self.opt_polytope_model.Gamma_min_softmax






    def constraint_func(self,q_des):

        self.opt_robot_model.urdf_transform(q_joints=q_des)
        self.pos_act = self.opt_robot_model.end_effector_position
        #print('Current position in optimization is',self.pos_act)
        #input('Wait here ')
        return -(norm(self.pos_act-self.pos_reference) - 1e-3)


    ### Constraints should be actual IK - Actual vs desrired - Cartesian pos

    ## NOrm -- || || < 1eps-
    def constraint_func_Gamma(self,q_des):

        self.opt_robot_model.urdf_transform(q_joints=q_des)
        #canvas_input.generate_axis()

        self.opt_polytope_model.generate_polytope_hyperplane(self.opt_robot_model,cartesian_dof_input = array([True,True,True,False,False,False]),
                                            qdot_min=self.qdot_min, qdot_max=self.qdot_max,
                                            cartesian_desired_vertices = self.desired_vertices,sigmoid_slope = self.sigmoid_slope)

        #print('Current objective in optimization Gamma is',self.opt_polytope_model.Gamma_min_softmax)
        return self.opt_polytope_model.Gamma_min_softmax

    def jac_func(self,q_des):
        from numpy import sum

        self.opt_robot_model.urdf_transform(q_joints=q_des)

        #self.opt_polytope_gradient_model.compute_polytope_gradient_parameters(self.opt_robot_model,self.opt_polytope_model)
        #self.opt_polytope_gradient_model.Gamma_hat_gradient(sigmoid_slope=1000)
        jac_output = self.opt_polytope_gradient_model.Gamma_hat_gradient(sigmoid_slope=self.sigmoid_slope)
        #print('jac_output',jac_output)

        #jac_output = sum(jac_output)
        return jac_output

    def hess_func(self,q_des):

        self.opt_robot_model.urdf_transform(q_joints=q_des)

        #self.opt_polytope_gradient_model.compute_polytope_gradient_parameters(self.opt_robot_model,self.opt_polytope_model)
        #self.opt_polytope_gradient_model.Gamma_hat_gradient(sigmoid_slope=1000)
        hess_output = self.opt_polytope_gradient_model.d_softmax_dq

        return hess_output
