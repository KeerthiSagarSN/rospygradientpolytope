# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:44:09 2022

@author: keerthi.sagar
"""
import scipy.optimize as sco

from numpy import array,pi,vstack,linspace,shape,zeros,hstack,transpose,matmul
from numpy.linalg import norm
from numpy.random import randn

from shapely.geometry import Polygon,LineString
from linearalgebra import isclose,V_unit
import matplotlib.pyplot as plt
from polytope_functions_2D import get_capacity_margin, get_polytope_hyperplane
from polytope_gradient_functions_2D import Gamma_hat_gradient_2D,hyperplane_gradient_2D, Gamma_hat_gradient_2D_dq
from gradient_functions_2D import normal_gradient,sigmoid_gradient
from WrenchMatrix import get_wrench_matrix
from robot_functions import sigmoid

from numpy import float64, average,matmul,dot

from visual_polytope import force_polytope_2D

import time

## Multiprocessing toolbox
import multiprocessing as mp

#from GenerateCanvas import GenerateCanvas
#import cProfile
#from linearalgebra import 
jac_output = mp.Array("f",zeros(shape=(2)))


plt.ion()
class OptimizationModel:

    def __init__(self):


        self.q_joints_input = None
        self.q_joints_opt = None
        #self.q_joints = None
        #self.obj_function = None
        self.function_opt = None

        self.gamma_input = None
        self.func_deriv = None

        #self.bnds = (-1.0,1.0)

        self.initial_x0 = None
        self.cons = None
        self.opt_robot_model = None
        self.opt_polytope_model = None
        self.opt_polytope_gradient_model = None
        self.qdot_min = None
        self.qdot_max = None
        self.cartesian_desired_vertices = None

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

        # Length of the base point frame 
        self.length_params = None
        self.height_params = None

        self.active_joints = None

        self.sigmoid_slope = None

        self.cartesian_dof_input = None

        self.step_size = None

        self.tol_value = None
        self.lower_bound = None

        self.cable_lines = {}

        

        


        
    ## Constraint equations - Cons1

    #ef constraint1(self):
    def gradient_descent_2D(f, grad_f, x0, learning_rate=0.1, tolerance=1e-6):
    
        from numpy.linalg import norm
        # Initial starting point

        x = x0
        previous_step_size = float("inf")
        
        while previous_step_size > tolerance:
            # Calculate the gradient at the current point
            grad = grad_f(x)
            
            # Update x in the direction of the negative gradient
            x = x - learning_rate * grad
            
            # Calculate the size of the last step
            previous_step_size = norm(learning_rate * grad)
        
        return x

    def test_gradient_2D(self):

        # Plot the obstacle with cable and base_points

        #ax = figure1.add_subplot(111, projection='2d')


        #global figure1
        #figure1 = plt.figure()

        #plt.ion()

        #plt.show()

        color_arr = ['k','r','b','c']

        self.active_joints = len(self.base_points)


        

        q_in_x = linspace(self.pos_bounds[0,0],self.pos_bounds[0,1],self.step_size)
        q_in_y = linspace(self.pos_bounds[1,0],self.pos_bounds[1,1],self.step_size)



        q_boundary_actual = zeros(shape = (len(q_in_x),len(q_in_y),2))
        q_boundary_actual[:,:,:] = -10000
        q_boundary_estimated = zeros(shape = (len(q_in_x),len(q_in_y),2))
        q_boundary_estimated[:,:,:] = -10000

        q_feasible = zeros(shape = (len(q_in_x),len(q_in_y),2))
        q_feasible[:,:,:] = -10000

        #q_total = zeros(shape = (len(q_in_x),len(q_in_y),2))
        q_total = zeros(shape=(len(q_in_x),len(q_in_y),2))

        q_total[:,:,:] = -10000
        q_infeasible = zeros(shape = (len(q_in_x),len(q_in_y),2))
        q_infeasible[:,:,:] = -10000

        CM_array_total_est = zeros(shape = (len(q_in_x),len(q_in_y)))

        CM_array_total_est[:,:] = -10000
        CM_array_total_actual = zeros(shape = (len(q_in_x),len(q_in_y)))
        CM_array_total_actual[:,:] = -10000

        CM_array_actual = zeros(shape = (len(q_in_x),len(q_in_y)))
        
        
        CM_array_est = zeros(shape = (len(q_in_x),len(q_in_y)))
        
        loop_counter = 0
        
        for i in range(len(q_in_x)):
        #for i in range(0,1): #len(q_in_x)):
            x_in = q_in_x[i]
            #x_in = 0.5
            for j in range(len(q_in_y)):
            #for j in range(0,1):
                

                print('loop_counter is',loop_counter)

                loop_counter += 1
                y_in = q_in_y[j]
                #y_in = 1.0



                q = array([x_in,y_in])




                #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
                
                Wm = zeros(shape=(2,self.active_joints))
                for k in range(len(self.base_points)):
                    cable_plt = array([[x_in,self.base_points[k,0]],[y_in,self.base_points[k,1]]])
                    Wm[0,k] = self.base_points[k,0] - x_in
                    Wm[1,k] = self.base_points[k,1] - y_in

                    #print('self.base_points',self.base_points[k,:])

                    #input('stop and check')

                    #Wm[0,k] = W[0,k]*((norm(W[:,k]))**(-1))
                    #Wm[1,k] = W[1,k]*((norm(W[:,k]))**(-1))

                    Wm[:,k] = V_unit(Wm[:,k])
                    #plt.plot(cable_plt[0,:],cable_plt[1,:],color = color_arr[k])
                    #plt.pause(0.01)
                    #print('cable number is:',k)
                #print('Wrench matrix is is',W)
                
                #W = W
                
                #print('Wrench matrix is here', W)
                #input('wait here')

                
                #input('stop here')
                #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
                #Wm = array([[-0.7071,-0.7071,-0.7071,-0.7071],[0.7071,0.7071,0.7071,0.7071]])
                
                #print('Wrench matrix is is',Wm)
                W = -Wm
                

                #W = W_n
                #JE = W
                #print('JE is',JE)
                #print('H is',H)
                #print('Wrench matrix is', J)

                h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
                    get_polytope_hyperplane(W,self.active_joints,self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.sigmoid_slope)
                    
                
                Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = \
                get_capacity_margin(W,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        self.active_joints,self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
                
                
                

                CM_array_total_actual[i,j] = Gamma_min

                CM_array_total_est[i,j] = Gamma_min_softmax

    def generate_workspace(self):

        # Plot the obstacle with cable and base_points

        #ax = figure1.add_subplot(111, projection='2d')


        #global figure1
        #figure1 = plt.figure()

        #plt.ion()

        #plt.show()

        color_arr = ['k','r','b','c']

        self.active_joints = len(self.base_points)


        

        q_in_x = linspace(self.pos_bounds[0,0]+0.002,self.pos_bounds[0,1]-0.002,self.step_size)
        q_in_y = linspace(self.pos_bounds[1,0]+0.002,self.pos_bounds[1,1]-0.002,self.step_size)



        q_boundary_actual = zeros(shape = (len(q_in_x),len(q_in_y),2))
        q_boundary_actual[:,:,:] = -10000
        q_boundary_estimated = zeros(shape = (len(q_in_x),len(q_in_y),2))
        q_boundary_estimated[:,:,:] = -10000

        q_feasible = zeros(shape = (len(q_in_x),len(q_in_y),2))
        q_feasible[:,:,:] = -10000

        #q_total = zeros(shape = (len(q_in_x),len(q_in_y),2))
        q_total = zeros(shape=(len(q_in_x),len(q_in_y),2))

        q_total[:,:,:] = -10000
        q_infeasible = zeros(shape = (len(q_in_x),len(q_in_y),2))
        q_infeasible[:,:,:] = -10000

        CM_array_total_est = zeros(shape = (len(q_in_x),len(q_in_y)))

        CM_array_total_est[:,:] = -10000
        CM_array_total_actual = zeros(shape = (len(q_in_x),len(q_in_y)))
        CM_array_total_actual[:,:] = -10000

        CM_array_actual = zeros(shape = (len(q_in_x),len(q_in_y)))
        
        
        CM_array_est = zeros(shape = (len(q_in_x),len(q_in_y)))
        
        loop_counter = 0
        first_iteration = True
        step_iter = self.step_size
        for i in range(len(q_in_x)):
        #for i in range(0,1): #len(q_in_x)):
            x_in = q_in_x[i]
            #x_in = 0.42
            #y_in = 0.30
            #
            for j in range(len(q_in_y)):
            #for j in range(0,1):
                

                #print('loop_counter is',loop_counter)

                loop_counter += 1
                y_in = q_in_y[j]
                #y_in += step_iter



                q = array([x_in,y_in])




                #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
                
                Wm = zeros(shape=(2,self.active_joints))
                for k in range(len(self.base_points)):
                    cable_plt = array([[x_in,self.base_points[k,0]],[y_in,self.base_points[k,1]]])
                    Wm[0,k] = self.base_points[k,0] - x_in
                    Wm[1,k] = self.base_points[k,1] - y_in

                    #print('self.base_points',self.base_points[k,:])

                    #input('stop and check')

                    #Wm[0,k] = W[0,k]*((norm(W[:,k]))**(-1))
                    #Wm[1,k] = W[1,k]*((norm(W[:,k]))**(-1))

                    Wm[:,k] = V_unit(Wm[:,k])
                    #plt.plot(cable_plt[0,:],cable_plt[1,:],color = color_arr[k])
                    #plt.pause(0.01)
                    #print('cable number is:',k)
                #print('Wrench matrix is is',W)
                
                #W = W
                
                #print('Wrench matrix is here', W)
                #input('wait here')

                
                #input('stop here')
                #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
                #Wm = array([[-0.7071,-0.7071,-0.7071,-0.7071],[0.7071,0.7071,0.7071,0.7071]])
                
                #print('Wrench matrix is is',Wm)
                W = -Wm
                

                #W = W_n
                #JE = W
                #print('JE is',JE)
                #print('H is',H)
                #print('Wrench matrix is', J)

                h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
                    get_polytope_hyperplane(W,self.active_joints,self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.sigmoid_slope)
                    
                
                Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = \
                get_capacity_margin(W,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        self.active_joints,self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
                
                Wu,Wn,Hess= get_wrench_matrix(q,self.length_params,self.height_params)

                #print('HEssian before gradient is',Hess)
                '''
                d_gamma_hat = Gamma_hat_gradient_2D(Wn,Hess,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                        p_minus_hat,Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat,\
                        self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
                '''
                CM_array_total_actual[i,j] = Gamma_min


                CM_array_total_est[i,j] = Gamma_min_softmax



                
                #print('Gamma_min is',Gamma_min)
                #print('Gamma_min softmax is',Gamma_min_softmax)
                #input('stop and test')
                
                #plt.scatter(self.base_points[:,0],self.base_points[:,1],color='k')
                #self.tol_value = 1e-3

                q_total[i,j,:] = q

                #dnk_dq_a = normal_gradient(Hess)
    
                #print('dnk_dq_a')
                #dnk_dq_a = Jy(x_in,y_in)
                #print('Hess',Hess)
                #dnk_dq_a = dnk_dq_a[:,:,1]

                #d_gamma_hat_a = d_gamma_hat[0]

                #print('Analytical gradient',d_gamma_hat_a)
                #print('Gamma_min',Gamma_min)

                #print('Gamma_min_softmax',Gamma_min_softmax)

                #input('check estimates')

                if not first_iteration:

                    

                    #d_gamma_hat_n = (Gamma_min_softmax - prev_gamma_min)/step_iter

                    #print('NUmerical gradient is',d_gamma_hat_n)
                    #print('Analytical gradient is',d_gamma_hat_a)

                    #print('Error')
                    #print(norm(d_gamma_hat_n- d_gamma_hat_a))

                    prev_gamma_min = Gamma_min_softmax


                    prev_n_k = n_k



                if first_iteration:

                    prev_n_k = n_k
                    prev_twist = Wn
                    prev_gamma_min = Gamma_min_softmax
                    first_iteration = False
                
                #if ((Gamma_min_softmax < self.tol_value)) and ((Gamma_min_softmax) > -self.lower_bound): Paper method
                if ((Gamma_min_softmax < self.tol_value)) and ((Gamma_min_softmax) > -self.lower_bound):
                    #print('inside WFW - estimated')
                    
                    print('Estimated boundary point')
                    
                    q_boundary_estimated[i,j,:] = q
                    #CM_array_est = vstack((CM_array_est,Gamma_min_softmax))
                    CM_array_est[i,j] = Gamma_min_softmax
                    #q_boundary_estimated = vstack((q_boundary_estimated,q))
                    #q_boundary_estimated[i,j,:] = q
                    #CM_array_est = vstack((CM_array_est,Gamma_min_softmax))
                    #CM_array_est[i,j] = Gamma_min_softmax
                    #print('Gamma_min_softmax')
                    #input('stop here')
                #print('Gamma_min is',Gamma_min)



                if ((Gamma_min < self.tol_value)) and ((Gamma_min > -self.lower_bound)): #Paper method
                #if ((Gamma_min < self.tol_value)) and ((Gamma_min > -self.lower_bound*1e-3)):
                    
                    print('Actual boundary pointttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt')
                    #print('Gamma_min',Gamma_min)
                    #print('The estimated CM at here is',Gamma_min_softmax)

                    #input('stop to thest')
                    '''
                    polytope_vertices, polytope_faces, facet_pair_idx, capacity_margin_faces, \
                        capacity_proj_vertex, polytope_vertices_est, polytope_faces_est, capacity_margin_faces_est, capacity_proj_vertex_est = \
                            force_polytope_2D(W,self.qdot_min, self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
                    '''
                    #figure1.canvas.draw()
                    #print('inside WFW - actual')
                    
                    #input('stop here')
                    #print('Feasiblyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                    
                    #scaling_factor = 0.005
                    #cartesian_desired_vertices_plt = (scaling_factor*self.cartesian_desired_vertices) + array([[x_in,y_in]])

                    #plt.plot(cartesian_desired_vertices_plt[:,0], cartesian_desired_vertices_plt[:,1],color = 'green')

                    #print('polytope_Vertices before',polytope_vertices)
                    #polytope_vertices = vstack((polytope_vertices,polytope_vertices[0,:]))
                    #polytope_vertices_plt = (scaling_factor*polytope_vertices) + array([[x_in,y_in]])

                    #print('polytope_Vertices after',polytope_vertices)
                    
                    #plt.plot(polytope_vertices_plt[:,0], polytope_vertices_plt[:,1],color = 'k')

                    #print('polytope_Vertices offset',polytope_vertices)
                    #print('polytope_faces',polytope_faces)

                    #print('facet_pair_idx',facet_pair_idx)
                    #print('capacity_margin_faces',capacity_margin_faces)
                    #print('capacity_proj_vertex',capacity_proj_vertex)
                    #desired_vertex_set = plt.scatter(cartesian_desired_vertices_plt[:,0], cartesian_desired_vertices_plt[:,1],color = 'k')
                    
                    '''
                    for i in range(len(n_k)):
                        
                        plt.plot([x_in,1*n_k[i,0]],[y_in,1*n_k[i,1]],color=color_arr[i])
                    '''
                    #plt.cla()

                    #figure1.canvas.flush_events()


                    #q_boundary_actual = vstack((q_boundary_actual,q))

                    q_boundary_actual[i,j,:] = q
                    #CM_array_actual = vstack((CM_array_actual,Gamma_min))
                    CM_array_actual[i,j] = Gamma_min
                    #input('testing here')



                    
                    
                    #plt.show()
                if (Gamma_min) > 0 :
                    print('feasible point')
                    #q_feasible = vstack((q_feasible,q))

                    q_feasible[i,j,:] = q

                    
                
                
                
                elif (Gamma_min) < -self.lower_bound :

                    print('infeasible point')
                    #q_infeasible = vstack((q_infeasible,q))
                    q_infeasible[i,j,:] = q
                

                   

                #plt.pause(0.01)
                #plt.gcf().clear()
        '''
        q_feasible = q_feasible[1:,:]
        q_infeasible = q_infeasible[1:,:]
        q_boundary_estimated = q_boundary_estimated[1:,:]
        q_boundary_actual = q_boundary_actual[1:,:]
        CM_array_actual = CM_array_actual[1:,:]

        CM_array_est = CM_array_est[1:,:]
        '''


        ef_total = q_total
        
        CM_estimated = CM_array_total_est

        
        #print('q_estimated is',q_estimated)
        #print('shape(0q_est)',shape(q_estimated))

        #global figure 7

        fig_1 = plt.figure()

        ax = plt.axes(projection='3d')



        CM_estimated_density = CM_estimated
        #print('CM_estimated',CM_estimated_density)
        #input('test here')

        #print('shape(CM_estimated density)',shape(CM_estimated_density))
        #print('shape of ef_total',shape(ef_total[:,:,0]))

        #X_dens,Y_dens = meshgrid(ef_total[:,:,0],ef_total[:,:,1])
        ''' 
        w = ax.plot_surface(ef_total[:,:,0], ef_total[:,:,1], CM_estimated_density,cmap='spring',alpha=0.4)
        # change the fontsize


        ax.set_xlabel('x [m]',fontsize=13)
        ax.set_ylabel('y [m]',fontsize=13)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        #ax.set_zlabel('$\hat{\gamma}$',fontsize=20)
        #ax.set_zlabel('$\hat{\gamma}$',rotation=145)
        ax.set_zlabel(r"$\hat{\gamma}$" + str(' [N]'),rotation=90,fontsize=16)
        
        ### Test gradient descent here
        #fig_2 = plt.figure()

        #ax2 = plt.axes()
        num_iterations = 1000
        learning_rate = 0.0001
        ### Good starting point x0 = 0.65, y0 = 0.72
        #x0 = 0.65
        #y0 = 0.72

        x0 = 0.005

        #x0 = 0.48
        y0 = 0.005
        y0 = 0.29
        #q = array([x_0,y_0])
        first_iteration = True

        x0_start = array([0.005,0.5,0.9])
        y0_start = array([0.1,0.1,0.9])

        color_arr = ['cyan','k','green',]
        color_arr = ['magenta','k','green','orange']
        color_arr = ['magenta','k','green','orange']

        color_arr = ['magenta','k','green','k']
        x_in = x0
        y_in = y0


        
        i0_plot = zeros(shape=(num_iterations))
        #len(x0_start)
        sigmoid_slope_arr = [10.0,20.0,30.0,50.0]
        error_plot_a = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
        error_plot_n = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
        z0_plot = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
        x0_plot = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
        y0_plot = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))

        
        #for lm in range(len(x0_start)):
        for lm in range(3,len(sigmoid_slope_arr )):
            self.sigmoid_slope = sigmoid_slope_arr[lm]
            #x0 = x0_start[lm]
            #x0 = 0.45 Good configuration to show
            #y0 = y0_start[lm]

            #y0 = 0.30 Good configuration to show
            x0 = 0.5
            y0 = 0.1

            x_in = x0
            y_in = y0
            for i in range(num_iterations):
                
                print('number of iterations',i)
                i0_plot[i] = i
                
                q = array([x0,y0])
                #q = array([x_in,y_in])

                x_in = x0
                y_in = y0
                #y_in += learning_rate

                test_joint = 1

                x0_plot[i,lm] = x_in
                y0_plot[i,lm] = y_in
                #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
                
                Wm = zeros(shape=(2,self.active_joints))
                for k in range(len(self.base_points)):
                    cable_plt = array([[x_in,self.base_points[k,0]],[y_in,self.base_points[k,1]]])
                    Wm[0,k] = self.base_points[k,0] - x_in
                    Wm[1,k] = self.base_points[k,1] - y_in

                    #print('self.base_points',self.base_points[k,:])

                    #input('stop and check')

                    #Wm[0,k] = W[0,k]*((norm(W[:,k]))**(-1))
                    #Wm[1,k] = W[1,k]*((norm(W[:,k]))**(-1))

                    Wm[:,k] = V_unit(Wm[:,k])
                    #plt.plot(cable_plt[0,:],cable_plt[1,:],color = color_arr[k])
                    #plt.pause(0.01)
                    #print('cable number is:',k)
                #print('Wrench matrix is is',W)
                
                #W = W
                
                #print('Wrench matrix is here', W)
                #input('wait here')

                
                #input('stop here')
                #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
                #Wm = array([[-0.7071,-0.7071,-0.7071,-0.7071],[0.7071,0.7071,0.7071,0.7071]])
                
                #print('Wrench matrix is is',Wm)
                W = Wm
                
                
                #W = W_n
                #JE = W
                #print('JE is',JE)
                #print('H is',H)
                #print('Wrench matrix is', J)




                h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
                    get_polytope_hyperplane(W,self.active_joints,self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.sigmoid_slope)
                    
                
                Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = \
                get_capacity_margin(W,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        self.active_joints,self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
                
                Wu,Wn,Hess= get_wrench_matrix(q,self.length_params,self.height_params)
                z0 = Gamma_min_softmax

                z0_plot[i,lm] = z0

                print('capacity margin is',z0)

                print('x position now is',x0)
                print('y position now is',y0)

                dn_dq = normal_gradient(Hess)            
                ax.scatter(x0,y0,z0,c=color_arr[k],s=4)   
                plt.pause(0.001)
                #plt.show()

                
                d_gamma_hat,d_LSE_dq ,d_LSE_dq_arr,d_gamma_max_dq,dn_dq = Gamma_hat_gradient_2D(Wn,Hess,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                        p_minus_hat,Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat,\
                        self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)

                #print('Gamma min is',Gamma_min)
                ##print('Gammaa_min softmax is',Gamma_min_softmax)
                #print('d_LSE_dq',d_LSE_dq)
                ##print('d_LSE_dq_arr',d_LSE_dq_arr)
                #print('d_gamma_max_dq',d_gamma_max_dq)
                x0 = x0 - learning_rate*d_gamma_hat[0]
                y0 = y0 - learning_rate*d_gamma_hat[1]
                #y0 += learning_rate
                #z0 = Gamma_min_softmax
                
                d_G_dq_a = -d_gamma_hat[test_joint]
                
                #dvk_dq_a = Hess[test_joint,2,:]

                #dvk_dq = Wn[:,2]
                #dn_dq_a = normal_gradient(Hess)

                #d_h_plus_dq_a = d_h_plus_dq
                #d_h_minus_dq_a = d_h_minus_dq
                
                #x_term = matmul(transpose(n_k[2,:]),Wn[:,0])

                #print('x_term is',x_term)

                #dx_dq_a = matmul(dn_dq[test_joint,2,:],Wn[:,0]) + matmul(transpose(n_k[2,:]),Hess[test_joint,0,:])

                

                #dsig_dq_a = sigmoid_gradient(2,0,dn_dq,n_k,Wn,Hess,test_joint,self.sigmoid_slope)
                #dsig_dq = sigmoid(x_term,self.sigmoid_slope)

                #dsig_dq_a = (dsig_dq*(1.0-dsig_dq))*dx_dq_a
                
                    
                if not first_iteration:
                    #ax2.scatter(i,z0,c='k',s=10)
                    #ax.scatter(x0,y0,z0,c='k',s=20)
                    d_G_dq_n = (Gamma_min_softmax-prev_Gamma_min_softmax)/(1.0*learning_rate)

                    #error_plot[i,lm] = ((d_G_dq_n - d_G_dq_a)/(d_G_dq_n))*100.0

                    #error_plot[i,lm] = ((d_G_dq_n - d_G_dq_a))

                    #error_plot_a[i,lm] = d_G_dq_a

                    #error_plot_n[i,lm] = d_G_dq_n

                    error_plot_n[i,lm] = ((d_G_dq_n - d_G_dq_a)/(d_G_dq_n))*100.0
                    #print('NUmerical gradient of gamma',(Gamma_min_softmax-prev_Gamma_min_softmax)/(1.0*learning_rate))
                    #print('ANalytical gradient',-d_gamma_hat)
                    prev_Gamma_min_softmax = Gamma_min_softmax
                    #input('test gradient')                    
                    #plt.pause(0.0001)
                    
                    #prev_n_k = n_k
                    #prev_h_plus = h_plus_hat
                    #prev_h_minus = h_minus_hat
                if first_iteration:
                    prev_Gamma_min_softmax = Gamma_min_softmax
                    #prev_n_k = n_k
                    #prev_h_plus = h_plus_hat
                    #prev_h_minus = h_minus_hat
                    #dsig_dq_prev = dsig_dq
                    #dvk_dq_prev = dvk_dq
                    #x_term_prev = x_term
                    first_iteration = False
                #input('test error here')
        '''   
        ##########3
        '''
        ax.plot(x0_plot[:,0],y0_plot[:,0],z0_plot[:,0],color=color_arr[0],marker='+', linestyle='dashed',label='x0:0.005, y0:0.1 ')
        ax.plot(x0_plot[:,1],y0_plot[:,1],z0_plot[:,1],color=color_arr[1],marker='+', linestyle='dashed',label='x0:0.5, y0:0.1 ')
        ax.plot(x0_plot[:,2],y0_plot[:,2],z0_plot[:,2],color=color_arr[2],marker='+', linestyle='dashed',label='x0:0.9, y0:0.9 ')
        #ax.plot(x0_plot[:,3],y0_plot[:,3],z0_plot[:,3],color=color_arr[3],marker='+', linestyle='dashed')
        ax.legend(loc="upper right",fontsize=25)
        ax2.plot(i0_plot,z0_plot[:,0],color=color_arr[0],linestyle='dashed',label='x0:0.005, y0:0.1 ')
        ax2.plot(i0_plot,z0_plot[:,1],color=color_arr[1],linestyle='dashed',label='x0:0.5, y0:0.1 ')
        ax2.plot(i0_plot,z0_plot[:,2],color=color_arr[2],linestyle='dashed',label='x0:0.9, y0:0.9 ')


        #ax2.plot(i0_plot,z0_plot[:,3],color=color_arr[3],linestyle='dashed')

        ax2.set_xlabel('Number of Iterations',prop={'size': 20})
        ax2.set_ylabel(r"$\hat{\gamma}$" + str(' [N]'),fontsize=13)
        plt.legend(loc="lower right")
        '''

        #ax2.plot(y0_plot[1:,0],error_plot_a[1:,0],color=color_arr[0],linestyle='dashed',label='Analytical Slope: 10')
        #ax2.plot(y0_plot[1:,0],error_plot_a[1:,1],color=color_arr[1],linestyle='dashed',label='Analytical Slope: 20')
        #ax2.plot(y0_plot[1:,0],error_plot_a[1:,2],color=color_arr[2],linestyle='dashed',label='Analytical Slope: 30')
        #ax2.plot(y0_plot[1:,0],error_plot_a[1:,3],color=color_arr[3],linestyle='dashed',label='Analytical Slope: 50')

        #input('second plot')
        '''
        ax2.plot(y0_plot[1:,0],error_plot_n[1:,0],color=color_arr[0],linestyle='solid',label='Numerical Slope: 10')
        ax2.plot(y0_plot[1:,0],error_plot_n[1:,1],color=color_arr[1],linestyle='solid',label='Numerical Slope: 20')
        ax2.plot(y0_plot[1:,0],error_plot_n[1:,2],color=color_arr[2],linestyle='solid',label='Numerical Slope: 30')
        ax2.plot(y0_plot[1:,0],error_plot_n[1:,3],color=color_arr[3],linestyle='solid',label='Numerical Slope: 50')
        ax2.set_xlabel('Y (m)')
        #ax2.set_ylabel(r"$\partial \hat{\gamma}$",fontsize=15)
        ax2.set_ylabel('Error $(\%)$',fontsize=15)

        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()
        ''' 

            

        return q_boundary_actual , q_boundary_estimated, q_feasible, q_infeasible, q_total,CM_array_actual, CM_array_est, CM_array_total_actual, CM_array_total_est
        



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


        ###### Plot all the obstacles here #######################################################
        #plt.show()
        input('fmin_opt')
        
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'r')
    
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]

        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)
        plt.plot(x,y,color = 'r')
        
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        

        ef = [0,0]
        cable_color = 'g'
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            self.cable_lines[c] = plt.plot(x,y,color = cable_color)
        plt.draw()

        

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
                {'type': 'ineq', 'fun': self.constraint_func_obstacles_2_c4},\
                {'type': 'ineq', 'fun': self.constr_function,'tol':1e-2})
        
        print('self.pos_bounds',self.pos_bounds[0,0])
        
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
                                         jac =self.jac_func, constraints = cons,method='SLSQP', \
                                             options={'disp': True,'maxiter':200})
        else:

            self.q_joints_opt = sco.minimize(fun = self.obj_function,  x0 = self.initial_x0,\
                                             jac = None, constraints = cons_cobyla,method='COBYLA', \
                                                 options={'disp': True,'maxiter':200})



        
        if (self.q_joints_opt.success):
            input('Success wait here to save plot') 

            ef = self.q_joints_opt.x
            for c in range(len(self.base_points)):
                x = [ef[0],self.base_points[c,0]]
                y = [ef[1],self.base_points[c,1]]
                self.cable_lines[c] = plt.plot(x,y,color = 'g')
                plt.scatter(self.base_points[c,0],self.base_points[c,1],color='k')
            
            #plt.scatter()
            plt.scatter(ef[0],ef[1],color='m')
            
            plt.savefig('CDPR_roi_optimization'+str('.png'),dpi=600)

        

            


    #hess = self.hess_func,
    #sco.check_grad(func = self.obj_function, grad =self.opt_polytope_gradient_model.d_gamma_hat \                                  , x0 = self.initial_x0, epsilon=1.4901161193847656e-08, direction='all', seed=None)
    #def constraint2(self):
    ## Obj
    def obj_function(self,q_des):
        self.active_joints = len(self.base_points)
        q = q_des
        #q = array([x_in,y_in])
        x0 = q[0]
        y0 = q[1]
        x_in = x0
        y_in = y0
        #y_in += learning_rate

        test_joint = 1

        #x0_plot[i,lm] = x_in
        #y0_plot[i,lm] = y_in
        #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
        
        Wm = zeros(shape=(2,self.active_joints))
        for k in range(len(self.base_points)):
            cable_plt = array([[x_in,self.base_points[k,0]],[y_in,self.base_points[k,1]]])
            Wm[0,k] = self.base_points[k,0] - x_in
            Wm[1,k] = self.base_points[k,1] - y_in

            #print('self.base_points',self.base_points[k,:])

            #input('stop and check')

            #Wm[0,k] = W[0,k]*((norm(W[:,k]))**(-1))
            #Wm[1,k] = W[1,k]*((norm(W[:,k]))**(-1))

            Wm[:,k] = V_unit(Wm[:,k])
            #plt.plot(cable_plt[0,:],cable_plt[1,:],color = color_arr[k])
            #plt.pause(0.01)
            #print('cable number is:',k)
        #print('Wrench matrix is is',W)
        
        #W = W
        
        #print('Wrench matrix is here', W)
        #input('wait here')

        
        #input('stop here')
        #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
        #Wm = array([[-0.7071,-0.7071,-0.7071,-0.7071],[0.7071,0.7071,0.7071,0.7071]])
        
        #print('Wrench matrix is is',Wm)
        W = Wm
        
        
        #W = W_n
        #JE = W
        #print('JE is',JE)
        #print('H is',H)
        #print('Wrench matrix is', J)




        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
            get_polytope_hyperplane(W,self.active_joints,self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.sigmoid_slope)
            
        
        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = \
        get_capacity_margin(W,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                self.active_joints,self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
        print('Capacity Margin for function eval is',Gamma_min_softmax)
        return -Gamma_min_softmax

    def constr_function(self,q_des):



        from numpy.linalg import norm
        from numpy import sum

        
        
        
        
        #print('Current constraint function distance in optimization is', norm(self.roi_center - q_des))
        
        '''
        
        obstacle_polytope_1 = self.obstacle_set[0]
        obstacle_polytope_2 = self.obstacle_set[1]
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')

        cable_color = 'g'
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            plt.plot(x,y,color = cable_color)

        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.gca().add_patch(roi_circle)
        plt.pause(0.0001)
        plt.cla()
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        '''
        
        ef = q_des
                      
        cable_color = 'g'
        for c in range(len(self.base_points)):
            line = self.cable_lines[c].pop(0)
            line.remove()
        
        
        for c in range(len(self.base_points)):
            x = [ef[0],self.base_points[c,0]]
            y = [ef[1],self.base_points[c,1]]
            self.cable_lines[c] = plt.plot(x,y,color = cable_color)
            
        plt.draw() 
        plt.pause(0.0001)
        

        return float64(norm(q_des-self.roi_center))
        

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
            
            #print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
            
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles 
        
        '''
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
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
        
        plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)  
        plt.pause(0.0001)
        plt.cla()
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        #input('wait here')
        '''
        

        
        return min_dist_obstacles - 0.005
    
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
            
            #print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
            
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles
        '''
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
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
        
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle)
        plt.pause(0.0001)
        plt.cla()
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        #input('wait here')
            
        '''

        
        return min_dist_obstacles - 0.005
    
    
    
    
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
            
            #print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
            
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles 
        '''
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
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

        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle)  
        plt.pause(0.0001)
        plt.cla()
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        #input('wait here')
        '''
        

        
        return min_dist_obstacles - 0.005
    
    
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
            
            #print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
        
        
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles 
        '''
        plt.xlabel('x[m]')
        plt.ylabel('y[m]') 
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
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        plt.pause(0.0001)
        plt.cla()
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        #input('wait here')
        '''
        

        
        return min_dist_obstacles -0.005
    
    
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
            
            #print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles
        
        '''
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
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

        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle)   
        plt.pause(0.0001)
        plt.cla()
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        #input('wait here')
            
        '''

        
        return min_dist_obstacles -0.005
    
    
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
            
            #print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
        
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
            #return min_dist_obstacles 
        
        '''
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')  
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

        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle)
        plt.pause(0.0001)
        plt.cla()
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        #input('wait here')
            
        

        '''
        return min_dist_obstacles - 0.005
    
    
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
            
            #print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
        
            
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0
        '''
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
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

        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle)   
        plt.pause(0.0001)
        plt.cla()
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        #input('wait here')
            
        '''

        
        return min_dist_obstacles  - 0.005
    
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
            
            #print('min_dist_obstacles',dist_cable_obs)
            min_dist_obstacles = dist_cable_obs
        if isclose(dist_cable_obs,0.0,1e-4):
            
            min_dist_obstacles = 0  
        '''
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
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

        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle)   
        plt.pause(0.0001)
        plt.cla()
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        #input('wait here')
        '''
        

        
        return min_dist_obstacles  - 0.005
    
    def constraint_func_obstacles_2(self,q_des):
        
        from numpy.linalg import norm
        ### Shapely Library

        
        
        ### M
        
        
        ### Cicrcle constraint equation
        
        cables = {}
        obstacles = {}
        min_dist_obstacles = 10000
        for i in range(len(self.base_points)):           
            
            #print('q_des',q_des)
            
            #print('base_points',self.base_points[i])
            
            cables[i] = LineString([q_des,self.base_points[i]])

            #for obs in range(len(self.obstacle_set)):
            obstacles = Polygon(self.obstacle_set[1])
            
            dist_cable_obs = obstacles.distance(cables[i])
            
            if isclose(dist_cable_obs,0.0,1e-4):
                
                min_dist_obstacles = 0
                return min_dist_obstacles - 0.001
            if dist_cable_obs < min_dist_obstacles:
                
                #print('min_dist_obstacles',dist_cable_obs)
                min_dist_obstacles = dist_cable_obs
            
        '''
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
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

        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        plt.pause(0.0001)
        plt.cla()
        roi_circle = plt.Circle((self.roi_center[0], self.roi_center[1]), 0.2, color='y', alpha=0.5)    
        plt.gca().add_patch(roi_circle) 
        #input('wait here')
            
        '''

        
        return min_dist_obstacles - 0.005


    ### Constraints should be actual IK - Actual vs desrired - Cartesian pos

    ## NOrm -- || || < 1eps-
    '''
    def constraint_func_Gamma(self,q_des):

        

        #print('Current objective in optimization Gamma is',self.opt_polytope_model.Gamma_min_softmax)
        
        
        
        
        return self.opt_polytope_model.Gamma_min_softmax
    '''
    def jac_func(self,q_des):
        #q = array([x0,y0])
        self.active_joints = len(self.base_points)
        q = q_des
        #q = array([x_in,y_in])
        x0 = q[0]
        y0 = q[1]
        x_in = x0
        y_in = y0
        #y_in += learning_rate

        #x0_plot[i,lm] = x_in
        #y0_plot[i,lm] = y_in
        #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
        
        Wm = zeros(shape=(2,self.active_joints))
        for k in range(len(self.base_points)):
            cable_plt = array([[x_in,self.base_points[k,0]],[y_in,self.base_points[k,1]]])
            Wm[0,k] = self.base_points[k,0] - x_in
            Wm[1,k] = self.base_points[k,1] - y_in


            #Wm[0,k] = W[0,k]*((norm(W[:,k]))**(-1))
            #Wm[1,k] = W[1,k]*((norm(W[:,k]))**(-1))

            Wm[:,k] = V_unit(Wm[:,k])
            #plt.plot(cable_plt[0,:],cable_plt[1,:],color = color_arr[k])
            #plt.pause(0.01)


        
        #W = W

        #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
        #Wm = array([[-0.7071,-0.7071,-0.7071,-0.7071],[0.7071,0.7071,0.7071,0.7071]])

        W = Wm
        
        
        #W = W_n
        #JE = W
 




        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
            get_polytope_hyperplane(W,self.active_joints,self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.sigmoid_slope)
            
        
        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = \
        get_capacity_margin(W,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                self.active_joints,self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
        
        Wu,Wn,Hess= get_wrench_matrix(q,self.length_params,self.height_params)
        #z0 = Gamma_min_softmax

        #z0_plot[i,lm] = z0
        

               
        #ax.scatter(x0,y0,z0,c=color_arr[k],s=4)   
        #plt.pause(0.001)
        #plt.show()

        threads = []
        for i_thread in range(2):
            thread = mp.Process(target=Gamma_hat_gradient_2D_dq,args=(Wn,Hess,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                p_minus_hat,Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat,\
                self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope,i_thread,jac_output))
            thread.start()
            threads.append(thread)
        
        # now wait for them all to finish
        for thread in threads:
            thread.join()
        
        
        


        return jac_output
