# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:44:09 2022

@author: keerthi.sagar
"""
import scipy.optimize as sco

from numpy import array,pi
from numpy.linalg import norm
from numpy.random import randn

from shapely.geometry import Polygon,LineString
from linearalgebra import isclose
import matplotlib.pyplot as plt
#from GenerateCanvas import GenerateCanvas
#import cProfile
#from linearalgebra import 

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
        #self.pos_reference = None
        
        self.pos_act = None
        
        
        ## Assign obstacle_set as a list of Shapely- Convex polytope objects here
        self.obstacle_set = None
        
        #self.obstacle_shapely_set = 
        self.analytical_solver = None
        
        # Base_points of the CDPT
        self.base_points = None
        
        ## Center of region of interest
        self.roi_center = None
        
        self.pos_bounds = None
    ## Constraint equations - Cons1

    #ef constraint1(self):



    ## Constraint equations - Cons2



    #def fmin_opt(self,robot_model, polytope_model, polytope_gradient_model,desired_vertices_inp,reference_pos,sigmoid_slope_inp,analytical_solver:bool):
    def fmin_opt(self,reference_pos):
        ### Function - func
        ## Initial point - x0
        ## args -
        ## method - SLQSQ
        ## jac = Jacobian - gradient of the

        ###
        #self.canvas_input_opt = GenerateCanvas()
        #self.canvas_input_opt.set_params(view_ang1=30,view_ang2=45,x_limits=[-20,20],y_limits=[-20,20],z_limits=[-20,20],axis_off_on = True)

        # Plot the obstacle with cable and base_points
        global figure1
        figure1 = plt.figure()
        '''
        
        self.sigmoid_slope = sigmoid_slope_inp


        self.desired_vertices = desired_vertices_inp
        self.opt_robot_model = robot_model
        self.opt_polytope_model = polytope_model
        self.opt_polytope_gradient_model = polytope_gradient_model

        self.q_joints_input = robot_model.q_joints


        self.qdot_min = polytope_model.qdot_min
        self.qdot_max = polytope_model.qdot_max
        
        self.initial_x0 = robot_model.q_joints
        
        self.opt_polytope_gradient_model.compute_polytope_gradient_parameters(self.opt_robot_model,self.opt_polytope_model)
        self.opt_polytope_gradient_model.Gamma_hat_gradient(sigmoid_slope=self.sigmoid_slope)
        '''
        
        self.pos_reference = reference_pos
        self.initial_x0 = reference_pos
        
        #self.obstacle_set = obstacle_set
        
        
        ### Desired vertex set
        '''
        self.desired_vertices[:,0] = self.desired_vertices[:,0] + self.pos_reference[0]
        self.desired_vertices[:,1] = self.desired_vertices[:,1] + self.pos_reference[1]
        self.desired_vertices[:,2] = self.desired_vertices[:,2] + self.pos_reference[2]
        '''
        
        #self.obj_function = polytope_model.Gamma_total
        #print('self.obj_function',self.obj_function)

        
        

        #self.initial_x0 = randn(6)
        #print('self.initial_x0 ',self.initial_x0 )
        #print('self.obj_function(robot_model.q_joints)',self.obj_function(robot_model.q_joints))
        #self.func_deriv = polytope_gradient_model.d_gamma_hat



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

        
        #print('Reference position is',self.pos_reference)



        #print('self.opt_polytope_gradient_model.d_gamma_hat',self.opt_polytope_gradient_model.d_gamma_hat)
        
        '''
        # Bounds created from the robot angles

        self.opt_bounds = self.opt_robot_model.q_joints_bounds
        '''
        #print('self.opt_bounds is',self.opt_bounds)

        ### Constraints
        '''
        cons = ({'type': 'ineq', 'fun': self.constraint_func},\
                {'type': 'ineq', 'fun': self.constraint_func_Gamma})
        '''
        ### Need to construct constraints in the fly dynamically rather than typing explicity - for cable and for obstacle
        cons = ({'type': 'ineq', 'fun': self.constraint_func_obstacles_1_c1},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_1_c2},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_1_c3},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_1_c4},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_2_c1},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_2_c2},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_2_c3},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_2_c4})
        
        print('self.pos_bounds',self.pos_bounds[0,0])
        input('wait here')
        cons_cobyla = ({'type': 'ineq', 'fun': self.constraint_func_obstacles_1_c1},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_1_c2},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_1_c3},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_1_c4},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_2_c1},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_2_c2},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_2_c3},\
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_2_c4},\
                    {'type': 'ineq', 'fun': lambda q_des: q_des[0] - self.pos_bounds[0,0]},\
                        {'type': 'ineq', 'fun': lambda q_des: self.pos_bounds[0,1] - q_des[0]},\
                            {'type': 'ineq', 'fun': lambda q_des: q_des[1] - self.pos_bounds[1,0]},\
                                {'type': 'ineq', 'fun': lambda q_des: self.pos_bounds[1,1] - q_des[1]})
            
        '''
        cons = ({'type': 'ineq', 'fun': lambda x:  self.q_joints_input[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
        '''
        ## jACOBIAN MATRIX OF THE OBJECTIVE FUNCTION

        # jac = self.opt_polytope_gradient_model.d_gamma_hat,

        ### WIth analyitacl gradient

        if self.analytical_solver:

            self.q_joints_opt = sco.minimize(fun = self.obj_function,  x0 = self.initial_x0,bounds = self.pos_bounds,\
                                         jac =None, constraints = cons,method='SLSQP', \
                                             options={'disp': True,'maxiter':5000})
        else:

            self.q_joints_opt = sco.minimize(fun = self.obj_function,  x0 = self.initial_x0,\
                                             jac = None, constraints = cons_cobyla,method='COBYLA', \
                                                 options={'disp': True,'maxiter':10000})



    

            


    #hess = self.hess_func,
    #sco.check_grad(func = self.obj_function, grad =self.opt_polytope_gradient_model.d_gamma_hat \                                  , x0 = self.initial_x0, epsilon=1.4901161193847656e-08, direction='all', seed=None)
    #def constraint2(self):
    ## Obj
    def obj_function(self,q_des):



        from numpy.linalg import norm
        from numpy import sum

        
        
        
        
        print('Current objective function in optimization is', norm(self.roi_center - q_des))
        
        
        
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        
        ef = q_des
        
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)
            
        plt.pause(0.05)
        plt.cla()
        

        return norm(self.roi_center - q_des)
        

            #plt.cla()

    

    ## 
    def constraint_func_obstacles_1_c1(self,q_des):
        
        from numpy.linalg import norm
        ### Shapely Library

        
        
        ### M
        
        
        ### Cicrcle constraint equation
        
        cables = {}
        obstacles = {}
        min_dist_obstacles = 10000
        #for i in range(len(self.base_points)):           
            
       
        
        cables = LineString([q_des,self.base_points[0]])

        #for obs in range(len(self.obstacle_set)):
        obstacles = Polygon(self.obstacle_set[0])
        
        dist_cable_obs = obstacles.distance(cables)
        

        if dist_cable_obs < min_dist_obstacles:
            
            print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
            
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles 
        
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        
        ef = q_des
        
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)
            
        plt.pause(0.05)
        plt.cla()
        #input('wait here')
            
        

        
        return min_dist_obstacles - 0.2
    
    def constraint_func_obstacles_1_c2(self,q_des):
        
        from numpy.linalg import norm
        ### Shapely Library

        
        
        ### M
        
        
        ### Cicrcle constraint equation
        
        cables = {}
        obstacles = {}
        min_dist_obstacles = 10000
        #for i in range(len(self.base_points)):           
            
       
        
        cables = LineString([q_des,self.base_points[1]])

        #for obs in range(len(self.obstacle_set)):
        obstacles = Polygon(self.obstacle_set[0])
        
        dist_cable_obs = obstacles.distance(cables)
        

        if dist_cable_obs < min_dist_obstacles:
            
            print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
            
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles
        
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        
        ef = q_des
        
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)
            
        plt.pause(0.05)
        plt.cla()
        #input('wait here')
            
        

        
        return min_dist_obstacles - 0.2
    
    
    
    
    def constraint_func_obstacles_1_c3(self,q_des):
        
        from numpy.linalg import norm
        ### Shapely Library

        
        
        ### M
        
        
        ### Cicrcle constraint equation
        
        cables = {}
        obstacles = {}
        min_dist_obstacles = 10000
        #for i in range(len(self.base_points)):           
            
       
        
        cables = LineString([q_des,self.base_points[2]])

        #for obs in range(len(self.obstacle_set)):
        obstacles = Polygon(self.obstacle_set[0])
        
        dist_cable_obs = obstacles.distance(cables)
        

        if dist_cable_obs < min_dist_obstacles:
            
            print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
            
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles 
        
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        
        ef = q_des
        
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)
            
        plt.pause(0.05)
        plt.cla()
        #input('wait here')
            
        

        
        return min_dist_obstacles - 0.2
    
    
    def constraint_func_obstacles_1_c4(self,q_des):
        
        from numpy.linalg import norm
        ### Shapely Library

        
        
        ### M
        
        
        ### Cicrcle constraint equation
        
        cables = {}
        obstacles = {}
        min_dist_obstacles = 10000
        #for i in range(len(self.base_points)):           
            
       
        
        cables = LineString([q_des,self.base_points[3]])

        #for obs in range(len(self.obstacle_set)):
        obstacles = Polygon(self.obstacle_set[0])
        
        dist_cable_obs = obstacles.distance(cables)
        

        if dist_cable_obs < min_dist_obstacles:
            
            print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
        
        
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles 
            
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        
        ef = q_des
        
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)
            
        plt.pause(0.05)
        plt.cla()
        #input('wait here')
            
        

        
        return min_dist_obstacles -0.2
    
    
    def constraint_func_obstacles_2_c1(self,q_des):
        
        from numpy.linalg import norm
        ### Shapely Library

        
        
        ### M
        
        
        ### Cicrcle constraint equation
        
        cables = {}
        obstacles = {}
        min_dist_obstacles = 10000
        #for i in range(len(self.base_points)):           
            
       
        
        cables = LineString([q_des,self.base_points[0]])

        #for obs in range(len(self.obstacle_set)):
        obstacles = Polygon(self.obstacle_set[1])
        
        dist_cable_obs = obstacles.distance(cables)
        
 
        if dist_cable_obs < min_dist_obstacles:
            
            print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles
            
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        
        ef = q_des
        
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)
            
        plt.pause(0.05)
        plt.cla()
        #input('wait here')
            
        

        
        return min_dist_obstacles -0.2
    
    
    def constraint_func_obstacles_2_c2(self,q_des):
        
        from numpy.linalg import norm
        ### Shapely Library

        
        
        ### M
        
        
        ### Cicrcle constraint equation
        
        cables = {}
        obstacles = {}
        min_dist_obstacles = 10000
        #for i in range(len(self.base_points)):           
            
       
        
        cables = LineString([q_des,self.base_points[1]])

        #for obs in range(len(self.obstacle_set)):
        obstacles = Polygon(self.obstacle_set[1])
        
        dist_cable_obs = obstacles.distance(cables)
        

        if dist_cable_obs < min_dist_obstacles:
            
            print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
        
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles 
            
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        
        ef = q_des
        
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)
            
        plt.pause(0.05)
        plt.cla()
        #input('wait here')
            
        

        
        return min_dist_obstacles - 0.2
    
    
    def constraint_func_obstacles_2_c3(self,q_des):
        
        from numpy.linalg import norm
        ### Shapely Library

        
        
        ### M
        
        
        ### Cicrcle constraint equation
        
        cables = {}
        obstacles = {}
        min_dist_obstacles = 10000
        #for i in range(len(self.base_points)):           
            
       
        
        cables = LineString([q_des,self.base_points[2]])

        #for obs in range(len(self.obstacle_set)):
        obstacles = Polygon(self.obstacle_set[1])
        
        dist_cable_obs = obstacles.distance(cables)
        

            #return min_dist_obstacles 
        if dist_cable_obs < min_dist_obstacles:
            
            print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
        
            
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        
        ef = q_des
        
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)
            
        plt.pause(0.05)
        plt.cla()
        #input('wait here')
            
        

        
        return min_dist_obstacles  - 0.2
    
    def constraint_func_obstacles_2_c4(self,q_des):
        
        from numpy.linalg import norm
        ### Shapely Library

        
        
        ### M
        
        
        ### Cicrcle constraint equation
        
        cables = {}
        obstacles = {}
        min_dist_obstacles = 10000
        #for i in range(len(self.base_points)):           
            
       
        
        cables = LineString([q_des,self.base_points[3]])

        #for obs in range(len(self.obstacle_set)):
        obstacles = Polygon(self.obstacle_set[1])
        
        dist_cable_obs = obstacles.distance(cables)
        

            #return min_dist_obstacles 
        if dist_cable_obs < min_dist_obstacles:
            
            print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0  
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        
        ef = q_des
        
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)
            
        plt.pause(0.05)
        plt.cla()
        #input('wait here')
            
        

        
        return min_dist_obstacles  - 0.2
    
    def constraint_func_obstacles_2(self,q_des):
        
        from numpy.linalg import norm
        ### Shapely Library

        
        
        ### M
        
        
        ### Cicrcle constraint equation
        
        cables = {}
        obstacles = {}
        min_dist_obstacles = 10000
        for i in range(len(self.base_points)):           
            
            print('q_des',q_des)
            print('q_des',q_des)
            print('base_points',self.base_points[i])
            
            cables[i] = LineString([q_des,self.base_points[i]])

            #for obs in range(len(self.obstacle_set)):
            obstacles = Polygon(self.obstacle_set[1])
            
            dist_cable_obs = obstacles.distance(cables[i])
            
            if isclose(dist_cable_obs,0.0,1e-4):
                
                min_dist_obstacles = 0
                return min_dist_obstacles - 0.001
            if dist_cable_obs < min_dist_obstacles:
                
                print('min_dist_obstacles',dist_cable_obs)
                min_dist_obstacles = dist_cable_obs
            
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        
        ef = q_des
        
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)
            
        plt.pause(0.05)
        plt.cla()
        #input('wait here')
            
        

        
        return min_dist_obstacles - 0.01


    ### Constraints should be actual IK - Actual vs desrired - Cartesian pos

    ## NOrm -- || || < 1eps-
    def constraint_func_Gamma(self,q_des):

        

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
