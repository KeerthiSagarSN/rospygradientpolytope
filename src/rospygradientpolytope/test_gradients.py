# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 21:33:06 2022

@author: keerthi.sagar
"""

import os
import sympy
import matplotlib
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager


#from ....stormkinpy.stormkinpy import visualbuild_v2

import sys
#sys.path.append('/Gihub_repos/stormkinpy/stormkinpy/')
#python -m /Github_repos.stormkinpy.stormkinpy.visualbuild_v2
#from __Github_repos__ import visualbuild_v2
## Import library from stormkinpy repository
#SET PYTHONPATH=”path/to/directory”

## Need to do relative python import for release
sys.path.insert(0,'C:/Users/keerthi.sagar/Dropbox/Github_repos/stormkinpy/stormkinpy')

sys.path.insert(0,'C:/Users/keerthi.sagar/Dropbox/Github_repos/stormkinpy/stormkinpy/URDF')
from visualbuild_v2 import line_plot,ScrewsPlot,revolute_joint,link_generate,spherical_joint,plane_plot,line_plot
from numpy import cross,hstack,nonzero,vstack,array,zeros,cross,any,where,logical_and,round,matmul,transpose,copy,argpartition
from math import isnan,pow
from numpy.linalg import norm

from numpy.linalg import svd,det

from linearalgebra import sigmoid_func,twolines_intersection,plane_plane_intersection,vectors_on_plane
from linearalgebra import V_unit,isclose
from pycapacity.robot import velocity_polytope_withfaces
from pycapacity.visual import plot_polytope_vertex,plot_polytope_faces

'''
from polytope_gradient_functions import gradient_LSE, LSE_func,cross_product_gradient_normalized,gradient_cross_product,gradient_cross_product_normalized
from polytope_gradient_functions import normal_hyperplane_gradient,sigmoid_gradient,sigmoid
from polytope_functions import get_Cartesian_polytope,get_reduced_hyperplane_parameters
from polytope_functions import get_gamma_hat, get_gamma,plot_polytope_3d,get_hyperplane_parameters
'''
from screwtheory_v6_lib import *

from scipy.spatial import ConvexHull

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import robot_functions
from robot_functions import sigmoid
'''
import polytope_functions
import sawyer_functions
import polytope as pc
'''


from GenerateCanvas import GenerateCanvas
from GenerateScrews import GenerateScrews
from PlotScrews import PlotScrews
from TransformURDF import TransformURDF
from PlotPrimitives import PlotPrimitives
#from PolytopeGradient import PolytopeGradient
import robot_functions 
from RobotModel import RobotModel
from numpy import pi,random,arange
from numpy.random import randn
#from PolytopeModel import PolytopeModel

from numpy.linalg import det

## Generate canvas
#canvas = GenerateCanvas()

## Generate figure and plot image here
#plt,ax1= canvas.generate_axis()

#canvas.set_params(plt,ax1,30,45,[-1,1],[-1,1],[-1,1],False)




#def test_hessian(limits,step):



class TestGradient:
    
    def __init__(self):
        
        self.test_robot_model = None
        self.test_polytope_model = None
        self.test_polytope_gradient = None
        self.q_joint = None
        self.joint = None
        self.cartesian_dof_input = None
        self.qdot_min = None
        self.qdot_max = None
        self.cartesian_desired_vertices = None

    
    def launch_robot(self,robot_name:chr,world_frame_input:chr,urdf_address,canvas_plot,end_effector_offset:list,
                   end_effector_position_offset:float, plot_screws_on,plot_primitives_on,plot_urdf_on):
        self.test_robot_model = RobotModel()
        
        self.test_robot_model.urdf_start(robot_name,world_frame_input,urdf_address,canvas_plot,end_effector_offset,
                   end_effector_position_offset, plot_screws_on,plot_primitives_on,plot_urdf_on)
        
        
        
        # Generating active joint angles randomly here
        self.q_joint = randn(self.test_robot_model.obj_robot.number_of_joints)
        self.q_joint = array([-0.3,0.3,-0.3,0.3,-0.1,0.1])
        self.test_robot_model.urdf_transform(q_joints=self.q_joint)
        
        # Choosing a random joint here
        
        self.joint= random.randint(self.test_robot_model.obj_robot.number_of_joints)
        self.joint = 3
        #joint=1
        
    
    def polytope_parameters(self,cartesian_dof_input,qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope):
        #self.test_polytope_model = PolytopeModel()
        #self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,cartesian_dof_input,qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope)
        
        self.cartesian_dof_input  = cartesian_dof_input
        self.qdot_min = qdot_min
        self.qdot_max = qdot_max
        self.deltaqq = self.qdot_max - self.qdot_min
        self.cartesian_desired_vertices = cartesian_desired_vertices
        self.sigmoid_slope = sigmoid_slope
        
    ## Function similar to Philip's polytope_test

        
    def test_hessian(self,limits,step):
               
        from numpy.random import randn

        test_domain = arange(limits[0], limits[1], step)
        
        
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        
               
                
        J = self.test_robot_model.jacobian_hessian

        error = []
        nm_tes=[]
        first_iteration = True
        for z in test_domain:
            J_last=J
            
            if first_iteration == False:
                self.q_joint[self.joint]=z
    
                self.test_robot_model.urdf_transform(self.q_joint)
                
                                
                J = self.test_robot_model.jacobian_hessian
            
            
                 
                numerical_gradient = ((J-J_last) / step)
                
                print('numerical gradient is')
                print(numerical_gradient)
                #nm_tes.append(numerical_gradient[2,3])
                
                #self.test_polytope_gradient.jacobian_gradient(self.joint)
                #analytical_gradient = self.test_polytope_gradient.d_Je_dq
                analytical_gradient = self.test_robot_model.Hessian[:,:,self.joint]
                print('analytical_gradient is')
                print(analytical_gradient)
                
                error.append( np.linalg.norm(numerical_gradient-analytical_gradient))
                print('error is')
                print(np.linalg.norm(numerical_gradient-analytical_gradient))          
                #input('wait here')
            first_iteration = False
       
        plt.plot(error[1:], 'r')
        plt.show()        
        max(error)


    def test_twist_gradient(self,limits,step):
               
        from numpy.random import randn
        from gradient_functions import twist_gradient
        from numpy.linalg import norm
        #from gradient_functions import twist_gradient

        test_domain = arange(limits[0], limits[1], step)
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        
        #self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,self.cartesian_dof_input,
        #                                                      self.qdot_min,self.qdot_max,self.cartesian_desired_vertices)
        
        #self.test_polytope_gradient.compute_polytope_gradient_parameters(self.test_robot_model,self.test_polytope_model)
        
                
        twist_1 = self.test_robot_model.jacobian_hessian[0:3,0]

        error = []
        nm_tes=[]
        for z in test_domain:
            twist_last=twist_1
            self.q_joint[self.joint]=z

            self.test_robot_model.urdf_transform(self.q_joint)
            
            #self.test_polytope_gradient.compute_polytope_gradient_parameters(self.test_robot_model,self.test_polytope_model)
            
            twist_1 = self.test_robot_model.jacobian_hessian[0:3,0]
            
            
            print('twist_1')
            print(twist_1)
        
            numerical_gradient = ((twist_1 -twist_last) / step)
                
            print('numerical gradient is')
            print(numerical_gradient)
            #nm_tes.append(numerical_gradient[2,3])
            
            
            analytical_gradient = twist_gradient(0,self.test_robot_model.Hessian,self.joint)
            print('analytical_gradient is')
            print(analytical_gradient)
            
            error.append(norm(numerical_gradient-analytical_gradient))
            print('error is')
            print(np.linalg.norm(numerical_gradient-analytical_gradient))          
                   
       
        plt.plot(error[1:], 'r')
        plt.show()        
        max(error)

    def test_cross_product_gradient(self,limits,step):
               
        from numpy.random import randn

        test_domain = arange(limits[0], limits[1], step)
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        
        self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,self.cartesian_dof_input,
                                                              self.qdot_min,self.qdot_max,self.cartesian_desired_vertices)
        
        self.test_polytope_gradient.compute_polytope_gradient_parameters(self.test_robot_model,self.test_polytope_model)
        
        
        twist_index_1 =1
        twist_index_2 = 4
        twist_1 = self.test_robot_model.jacobian_hessian[:,twist_index_1][0:3]
        twist_2 = self.test_robot_model.jacobian_hessian[:,twist_index_2][0:3]
        
        cross_product_test = cross(twist_1,twist_2)

        error = []
        nm_tes=[]
        for z in test_domain:
            
            cross_product_test_last=cross_product_test
            self.q_joint[self.joint]=z

            self.test_robot_model.urdf_transform(self.q_joint)
            
            self.test_polytope_gradient.compute_polytope_gradient_parameters(self.test_robot_model,self.test_polytope_model)
            
            twist_1 = self.test_robot_model.jacobian_hessian[:,twist_index_1][0:3]
            twist_2 = self.test_robot_model.jacobian_hessian[:,twist_index_2][0:3]
            
            cross_product_test = cross(twist_1,twist_2)
            
            
            print('twist_1')
            print(twist_1)
        
            numerical_gradient = ((cross_product_test -cross_product_test_last) / step)
                
            print('numerical gradient is')
            print(numerical_gradient)
            #nm_tes.append(numerical_gradient[2,3])
            
            
            analytical_gradient = self.test_polytope_gradient.cross_product_gradient(twist_index_1,twist_index_2,
                                                                                     self.joint)
            print('analytical_gradient is')
            print(analytical_gradient)
            
            error.append( np.linalg.norm(numerical_gradient-analytical_gradient))
            print('error is')
            print(np.linalg.norm(numerical_gradient-analytical_gradient))          
                   
       
        plt.plot(error[1:], 'r')
        plt.show()        
        max(error)
    
    


    def test_cross_product_norm_gradient(self,limits,step):
               
        from numpy.random import randn
        from numpy.linalg import norm
        from gradient_functions import cross_product_norm_gradient

        test_domain = arange(limits[0], limits[1], step)
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        '''
        self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,self.cartesian_dof_input,
                                                              self.qdot_min,self.qdot_max,self.cartesian_desired_vertices)
        
        self.test_polytope_gradient.compute_polytope_gradient_parameters(self.test_robot_model,self.test_polytope_model)
        '''
        
        twist_index_1 =2
        twist_index_2 = 5
        twist_1 = self.test_robot_model.jacobian_hessian[:,twist_index_1][0:3]
        twist_2 = self.test_robot_model.jacobian_hessian[:,twist_index_2][0:3]
        
        cross_product_norm_test = norm(cross(twist_1,twist_2))

        error = []
        nm_tes=[]
        for z in test_domain:
            
            cross_product_norm_test_last=cross_product_norm_test
            self.q_joint[self.joint]=z

            self.test_robot_model.urdf_transform(self.q_joint)
            
            #self.test_polytope_gradient.compute_polytope_gradient_parameters(self.test_robot_model,self.test_polytope_model)
            
            twist_1 = self.test_robot_model.jacobian_hessian[:,twist_index_1][0:3]
            twist_2 = self.test_robot_model.jacobian_hessian[:,twist_index_2][0:3]
            
            cross_product_norm_test = norm(cross(twist_1,twist_2))
            
            
            print('twist_1')
            print(twist_1)
        
            numerical_gradient = ((cross_product_norm_test -cross_product_norm_test_last) / step)
                
            print('numerical gradient is')
            print(numerical_gradient)
            #nm_tes.append(numerical_gradient[2,3])
            
            
            analytical_gradient = cross_product_norm_gradient(twist_index_1,twist_index_2, self.test_robot_model.jacobian_hessian,\
                                                              self.test_robot_model.Hessian,\
                                                                                     self.joint)
            print('analytical_gradient is')
            print(analytical_gradient)
            
            error.append( np.linalg.norm(numerical_gradient-analytical_gradient))
            print('error is')
            print(np.linalg.norm(numerical_gradient-analytical_gradient))          
                   
       
        plt.plot(error[1:], 'r')
        plt.show()        
        max(error)
        

    def test_normal_gradient(self,limits,step):
               
        from numpy.random import randn
        from numpy.linalg import norm
        from linearalgebra import V_unit
        from gradient_functions import normal_gradient

        test_domain = arange(limits[0], limits[1], step)
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        
        '''
        self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,self.cartesian_dof_input,
                                                              self.qdot_min,self.qdot_max,self.cartesian_desired_vertices)
        
        self.test_polytope_gradient.compute_polytope_gradient_parameters(self.test_robot_model,self.test_polytope_model)
        '''
        
        twist_index_1 =0
        twist_index_2 = 3
        twist_1 = self.test_robot_model.jacobian_hessian[0:3,twist_index_1]
        twist_2 = self.test_robot_model.jacobian_hessian[0:3,twist_index_2]
        
        norm_gradient_test =  V_unit(cross(twist_1,twist_2))

        error = []
        nm_tes=[]
        first_iteration = True
        for z in test_domain:
            
            norm_gradient_test_last = norm_gradient_test
                         
            self.q_joint[self.joint]=z

            self.test_robot_model.urdf_transform(self.q_joint)
            
            #self.test_polytope_gradient.compute_polytope_gradient_parameters(self.test_robot_model,self.test_polytope_model)
            
            twist_1 = self.test_robot_model.jacobian_hessian[0:3,twist_index_1]
            twist_2 = self.test_robot_model.jacobian_hessian[0:3,twist_index_2]
            
            norm_gradient_test = V_unit(cross(twist_1,twist_2))
            
            
            print('norm_gradient_test')
            print(norm_gradient_test)
            
            print('norm_gradient_test_last')
            print(norm_gradient_test_last)
            
            
            if first_iteration == False:   
                numerical_gradient = ((norm_gradient_test -norm_gradient_test_last) / step)
            
                first_iteration = False
                print('numerical gradient is')
                print(numerical_gradient)
                #nm_tes.append(numerical_gradient[2,3])
                
                #J = test_polytope_gradient.r
                
                analytical_gradient = normal_gradient(twist_index_1,twist_index_2,self.test_robot_model.jacobian_hessian,\
                                                      self.test_robot_model.Hessian,self.joint)
                
                #analytical_gradient = gradient_cross_product_normalized(J[0:3,twist_index_1],J[0:3,twist_index_2], H[0:3,twist_index_1,self.joint],H[0:3,twist_index_2,self.joint])
                
                #analytical_gradient = gradient_cross_product_normalized(twist_1,twist_2,)
                print('analytical_gradient is')
                print(analytical_gradient)
                
                error.append( np.linalg.norm(numerical_gradient-analytical_gradient))
                print('error is')
                print(np.linalg.norm(numerical_gradient-analytical_gradient))          
            first_iteration = False       
       
        plt.plot(error[1:], 'r')
        plt.show()        
        max(error)







    def test_sigmoid_gradient(self,limits,step):
               
        from numpy.random import randn,randint
        from numpy.linalg import norm
        from linearalgebra import V_unit
        from numpy import matmul, transpose,cross,isclose
        from robot_functions import sigmoid
        from gradient_functions import sigmoid_gradient
        from polytope_functions import get_polytope_hyperplane

        test_domain = arange(limits[0], limits[1], step)
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        
        
        
        ## Chossing a random
        
        
        
        
        
        
        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
            get_polytope_hyperplane(self.test_robot_model.jacobian_hessian,self.test_robot_model.active_joints,\
                                    self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
        
        
        test_index = randint(0,len(Nmatrix))
        
        twist_index_1 = Nmatrix[test_index,0]
        twist_index_2 = Nmatrix[test_index,1]
        twist_index_projected = Nnot[test_index,0]
        
        twist_1 = self.test_robot_model.jacobian_hessian[0:3,twist_index_1]
        twist_2 = self.test_robot_model.jacobian_hessian[0:3,twist_index_2]
        twist_projected = self.test_robot_model.jacobian_hessian[0:3, twist_index_projected]
        
        print('norm is norm(cross(twist_1,twist_2)) ##',norm(cross(twist_1,twist_2)))
        while (self.joint == twist_index_1) or (self.joint == twist_index_2) or (self.joint == twist_index_projected) or (isclose(norm(cross(twist_1,twist_2)),0.0)):
            # Choosing a random joint here
            self.joint= random.randint(self.test_robot_model.obj_robot.number_of_joints - 1)
            test_index = randint(0,len(Nmatrix))
            
            twist_1 = self.test_robot_model.jacobian_hessian[0:3,twist_index_1]
            twist_2 = self.test_robot_model.jacobian_hessian[0:3,twist_index_2]
            twist_projected = self.test_robot_model.jacobian_hessian[0:3, twist_index_projected]
        
        print('I have selected properly the twists #########################')
        
        n = cross(twist_1,twist_2)
        
        
        
        n_T_vk = matmul(transpose(n),twist_projected)
        
        sigmoid_gradient_test =  sigmoid(n_T_vk,100.0)

        error = []
        nm_tes=[]
        
        first_iteration = True
        for z in test_domain:
            
            sigmoid_gradient_test_last = sigmoid_gradient_test
            
            if not first_iteration:          
            
            
                self.q_joint[self.joint]=z
    
                self.test_robot_model.urdf_transform(self.q_joint)
                


                h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
                    get_polytope_hyperplane(self.test_robot_model.jacobian_hessian,self.test_robot_model.active_joints,\
                                            self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)

                
                twist_index_1 = Nmatrix[test_index,0]
                twist_index_2 = Nmatrix[test_index,1]
                twist_index_projected = Nnot[test_index,0]
                
                
                
                
                twist_1 = self.test_robot_model.jacobian_hessian[0:3,twist_index_1]
                twist_2 = self.test_robot_model.jacobian_hessian[0:3,twist_index_2]
                twist_projected = self.test_robot_model.jacobian_hessian[0:3, twist_index_projected]
                
                n = cross(twist_1,twist_2)
                

                
                n_T_vk = matmul(transpose(n),twist_projected)
                
                
                sigmoid_gradient_test =  sigmoid(n_T_vk,100.0)
                

        
            
                numerical_gradient = ((sigmoid_gradient_test -sigmoid_gradient_test_last) / step)
                print('numerical_gradient is')
                print(numerical_gradient)

                #nm_tes.append(numerical_gradient[2,3])
                
                #J = test_polytope_gradient.r
 
                analytical_gradient = sigmoid_gradient(twist_index_1,twist_index_2,twist_index_projected,\
                                                       self.test_robot_model.jacobian_hessian,self.test_robot_model.Hessian,self.joint,self.sigmoid_slope)
                  
                #analytical_gradient = gradient_cross_product_normalized(J[0:3,twist_index_1],J[0:3,twist_index_2], H[0:3,twist_index_1,self.joint],H[0:3,twist_index_2,self.joint])
                
                #analytical_gradient = gradient_cross_product_normalized(twist_1,twist_2,)
                print('analytical_gradient is')
                print(analytical_gradient)
                
                error.append( np.linalg.norm(numerical_gradient-analytical_gradient))
                print('error is')
                error1 = np.linalg.norm(numerical_gradient-analytical_gradient)
                print(error1)  
                if math.isnan(error1):
                    input('wait now')
            first_iteration = False
           
        plt.plot(error[1:], 'r')
        plt.show()        
        max(error)
        
        
    

    def test_hyperplane_gradient(self,limits,step):
               
        from numpy.random import randn
        from numpy.linalg import norm
        from linearalgebra import V_unit
        from copy import deepcopy
        from numpy import shape
        from polytope_functions_philip import get_hyperplane_parameters
        
        from polytope_functions import get_polytope_hyperplane
        
        from polytope_gradient_functions import hyperplane_gradient

        test_domain = arange(limits[0], limits[1], step)
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        
        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
            get_polytope_hyperplane(self.test_robot_model.jacobian_hessian,self.test_robot_model.active_joints,self.cartesian_dof_input,\
                                    self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
        
        
        d_h_plus_dq, d_h_minus_dq, dn_dq = hyperplane_gradient(self.test_robot_model.jacobian_hessian, self.test_robot_model.Hessian, n_k, Nmatrix, Nnot,\
                                                               h_plus_hat, h_minus_hat, p_plus_hat, p_minus_hat, self.qdot_min, self.qdot_max, self.joint, self.sigmoid_slope)

        
        
        
        
        h_plus_gradient_test_2 =  h_plus_hat
        
        n, hplus, hminus, d_n_dq, d_hplus_dq, d_hminus_dq = get_hyperplane_parameters(self.test_robot_model.jacobian_hessian[0:3,:], self.test_robot_model.Hessian, self.deltaqq,self.sigmoid_slope)
        #h_minus_gradient_test =   self.test_polytope_model.h_minus
        
        h_plus_gradient_test = hplus
        

        error = []
        error2 = []
        nm_tes=[]
        first_iteration = True
        for z in test_domain:
            
            h_plus_gradient_test_last = h_plus_gradient_test
            
            h_plus_gradient_test_2_last = h_plus_gradient_test_2
            #h_minus_gradient_test_last = h_minus_gradient_test
            
            if not first_iteration:
                self.q_joint[self.joint]=z
                
                #print(' self.joint')
                #print( self.joint)
    
                self.test_robot_model.urdf_transform(self.q_joint)
                
                
                h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
                    get_polytope_hyperplane(self.test_robot_model.jacobian_hessian,self.test_robot_model.active_joints,self.cartesian_dof_input,\
                                            self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
                
                
                
                d_h_plus_dq, d_h_minus_dq, dn_dq = hyperplane_gradient(self.test_robot_model.jacobian_hessian, self.test_robot_model.Hessian, n_k, Nmatrix, Nnot,\
                                                       h_plus_hat, h_minus_hat, p_plus_hat, p_minus_hat, self.qdot_min, self.qdot_max, self.joint, self.sigmoid_slope)



                
                #h_minus_gradient_test = self.test_polytope_model.h_minus
                
                
                #print(' h_plus')
                #print( h_plus_gradient_test)
                
                #print(' h_minus')
                #print( h_minus_gradient_test)
                
                #print(' h_plus')
                #print( h_plus_gradient_test_last)
                
                #print(' h_minus_last')
                #print( h_minus_gradient_test_last)
        
                n, hplus, hminus, d_n_dq, d_hplus_dq, d_hminus_dq = get_hyperplane_parameters(self.test_robot_model.jacobian_hessian[0:3,:], self.test_robot_model.Hessian, self.deltaqq, self.sigmoid_slope)
                
                
                #h_plus_gradient_test = self.test_polytope_model.h_plus
                
                h_plus_gradient_test = hplus
                
                
                
                #print( h_plus_gradient_test)
                
                
                #print(' hplus philip',hplus)
                h_plus_gradient_test_2 = h_plus_hat
                
                #print(' h_plus keerthi',h_plus_hat)
                
                #input('stop here')


                numerical_gradient = ((h_plus_gradient_test -h_plus_gradient_test_last) / step)
                
                numerical_gradient_2 = ((h_plus_gradient_test_2 -h_plus_gradient_test_2_last) / step)
                
                #numerical_gradient = ((h_minus_gradient_test -h_minus_gradient_test_last) / step)
                
                
               

                #print(numerical_gradient)
                #nm_tes.append(numerical_gradient[2,3])
                
                #J = test_polytope_gradient.r
                
                print('d_n_dq is',d_n_dq[:,:,self.joint])
                #print('len(d_n_dq) is',shape(d_n_dq[:,:,self.joint]))
                
                
                analytical_gradient = d_hplus_dq[:,self.joint]
                
                
                
                
                #analytical_gradient = gradient_cross_product_normalized(J[0:3,twist_index_1],J[0:3,twist_index_2], H[0:3,twist_index_1,self.joint],H[0:3,twist_index_2,self.joint])
                
                #analytical_gradient = gradient_cross_product_normalized(twist_1,twist_2,)
                #print('hplus analtical is')
                
                #print('hplus analtical is')
                #print(analytical_gradient)
                
                
                #self.test_polytope_gradient.hyperplane_gradient(self.joint)
                analytical_gradient_2 = d_h_plus_dq
                
                print('d_n_dq keerhti is ', dn_dq)
                
                #input('wait here')
                print('hplus - Keerthi', h_plus_hat)
                print('hplus - Philip',hplus)
                
                
                print('numerical gradient is',numerical_gradient)
                print('numerical gradient_2 is',numerical_gradient_2)
                
                print('analytical_gradient',analytical_gradient)
                print('analytical_gradient_2',analytical_gradient_2)
                input('wait here')
                
                
                error.append( max(numerical_gradient-analytical_gradient))
                print('error is')
                print(np.linalg.norm(numerical_gradient-analytical_gradient))
                
                error2.append( max(numerical_gradient_2-analytical_gradient_2))
                print('error2 is')
                print(np.linalg.norm(numerical_gradient_2-analytical_gradient_2))
                #input('wait here')
            first_iteration = False
                   
       
        plt.plot(error[1:], 'r')
        plt.plot(error2[1:], 'b')
        plt.show()        
        max(error)



    def test_gamma_gradient(self,limits,step):
               
        from numpy.random import randn
        from numpy.linalg import norm
        from linearalgebra import V_unit
        from copy import deepcopy
        from numpy import amax
        from polytope_functions import get_hyperplane_parameters,get_gamma

        test_domain = arange(limits[0], limits[1], step)
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        print('self.joint',self.joint)
        self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,self.cartesian_dof_input,
                                                              self.qdot_min,self.qdot_max,self.cartesian_desired_vertices)
        
        self.test_polytope_gradient.compute_polytope_gradient_parameters(self.test_robot_model,self.test_polytope_model)
        
        deltaqq = self.test_polytope_model.deltaqq
        qdot_min_input = array([0.3,0.3,0.3,0.3,0.3,0.3])
        qdot_max_input = array([0.6,0.6,0.6,0.6,0.6,0.6])
        
        desired_vertices = array([[0.20000, 0.50000, 0.50000],
                         [0.50000, -0.10000, 0.50000],
                         [0.50000, 0.50000, -0.60000],
                         [0.50000, -0.10000, -0.60000],
                         [-0.30000, 0.50000, 0.50000],
                         [-0.30000, -0.10000, 0.50000],
                         [-0.30000, 0.50000, -0.60000],                         
                         [-0.30000, -0.10000, -0.60000]])
        
        Gamma_plus, Gamma_minus, d_Gamma_plus_dq, d_Gamma_minus_dq = get_gamma(self.test_robot_model.jacobian_hessian[0:3,:], \
                                                                                                  self.test_robot_model.Hessian, qdot_max = qdot_max_input, \
                                                                                                      qdot_min = qdot_min_input, vertices = desired_vertices,sigmoid_slope = 100)
        
            
        self.test_polytope_gradient.Gamma_gradient(self.joint,sigmoid_slope=100)
        Gamma_gradient_test =  self.test_polytope_model.Gamma_plus_flat
        
        Gamma_gradient_test_2 = Gamma_plus
        
        #print('dn_dq keerthi iss:',self.test_polytope_gradient.dn_dq)
        
        #n, hplus, hminus, d_n_dq, d_hplus_dq, d_hminus_dq = get_hyperplane_parameters(self.test_robot_model.jacobian_hessian[0:3,:], self.test_robot_model.Hessian, deltaqq)
        #h_minus_gradient_test =   self.test_polytope_model.h_minus
        #print('d_n_dq philip iss:',d_n_dq)
        #h_plus_gradient_test = hplus
        

        error = []
        error2 = []
        nm_tes=[]
        first_iteration = True
        for z in test_domain:
            

            
            
            #h_plus_gradient_test_2_last = h_plus_gradient_test_2
            #h_minus_gradient_test_last = h_minus_gradient_test
            
            if not first_iteration:
                
                Gamma_gradient_test_last = Gamma_gradient_test
                Gamma_gradient_test_last_2 = Gamma_gradient_test_2
                self.q_joint[self.joint]=z
                
                #print(' self.joint')
                #print( self.joint)
    
                self.test_robot_model.urdf_transform(self.q_joint)
                self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,self.cartesian_dof_input,
                                                                  self.qdot_min,self.qdot_max,self.cartesian_desired_vertices)
                
                self.test_polytope_gradient.compute_polytope_gradient_parameters(self.test_robot_model,self.test_polytope_model)
                
                self.test_polytope_gradient.Gamma_gradient(self.joint,sigmoid_slope=100)
                
                Gamma_gradient_test =  self.test_polytope_model.Gamma_plus_flat
                
                
                Gamma_plus, Gamma_minus, d_Gamma_plus_dq, d_Gamma_minus_dq = get_gamma(self.test_robot_model.jacobian_hessian[0:3,:], \
                                                                                                  self.test_robot_model.Hessian, qdot_max = qdot_max_input, \
                                                                                          qdot_min = qdot_min_input, vertices = desired_vertices,sigmoid_slope = 100)
                    
                #print('dn_dq keerthi iss:',self.test_polytope_gradient.dn_dq)
                #n, hplus, hminus, d_n_dq, d_hplus_dq, d_hminus_dq = get_hyperplane_parameters(self.test_robot_model.jacobian_hessian[0:3,:], self.test_robot_model.Hessian, deltaqq)
                #h_minus_gradient_test =   self.test_polytope_model.h_minus
                #print('d_n_dq philip iss:',d_n_dq[:,:,self.joint])
                        
                
                Gamma_gradient_test_2 = Gamma_plus
                    
                    
                numerical_gradient = ((Gamma_gradient_test - Gamma_gradient_test_last) / step)
                
                numerical_gradient_2 = ((Gamma_gradient_test_2 - Gamma_gradient_test_last_2) / step)
                
                
                #self.test_polytope_gradient.Gamma_gradient(self.joint,100)
                
                #### negative sign i dont know why else 
                analytical_gradient = self.test_polytope_gradient.d_Gamma_plus_flat
                #input('wait here')
                
                
                analytical_gradient_2 = d_Gamma_plus_dq[:,:,self.joint]
                
                
                #print('Gamma_plus',Gamma_plus)
                #('Gamma_plus_keerthi',Gamma_gradient_test)
                
                
                print('numerical_gradient',numerical_gradient)
                
                print('numerical_gradient_2',numerical_gradient_2)
                
                print('analytical_gradient',analytical_gradient)
                print('analytical_gradient_2',analytical_gradient_2)
                
                #input('wait here')
                
                #print('numerical_gradient_2',numerical_gradient_2)
                #print('analytical_gradient_2',analytical_gradient_2)
                #input('wait here')
                

                
                error.append( amax(numerical_gradient-analytical_gradient))
                #print('error is')
                #print(np.linalg.norm(numerical_gradient-analytical_gradient))
                
                error2.append( amax(numerical_gradient_2 - analytical_gradient_2))
                #print(np.linalg.norm(numerical_gradient_2-analytical_gradient_2))
                

                
                #print('error',error)
                #print('error2',error2)
                
                #input('wait here')
            first_iteration = False

       
        plt.plot(error[1:], 'r')
        plt.plot(error2[1:], 'b')
       
        plt.show()        
        #max(error)



    def test_gamma_hat_gradient(self,limits,step):
               
        from numpy.random import randn
        from numpy.linalg import norm
        from linearalgebra import V_unit
        from copy import deepcopy
        from numpy import amax,max,exp,dot,matmul,ones
        from polytope_functions_philip import get_hyperplane_parameters,get_gamma,get_gamma_hat
        from polytope_functions import get_polytope_hyperplane,get_capacity_margin
        
        from polytope_gradient_functions import Gamma_hat_gradient_joint
        
        from robot_functions import exp_sum,exp_normalize,smooth_max_gradient
        from numpy import unravel_index,argmax,min
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        fig_ax = plt.figure()
        ax_poly= fig_ax.gca(projection='3d')
        

        test_domain = arange(limits[0], limits[1], step)
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        #print('self.joint',self.joint)
        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
            get_polytope_hyperplane(self.test_robot_model.jacobian_hessian,self.test_robot_model.active_joints,self.cartesian_dof_input,
                                                              self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
        
        
        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat = get_capacity_margin(self.test_robot_model.jacobian_hessian,\
                                                    n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,self.test_robot_model.active_joints,\
                                                        self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
        
        '''
        gamma_hat, d_gamma_hat_dq, Gamma_all_max = get_gamma_hat(self.test_robot_model.jacobian_hessian[0:3,:], \
                                                                 self.test_robot_model.Hessian, \
                                                                     qdot_max = qdot_max_input, \
                                                                         qdot_min = qdot_min_input, \
                                                                             vertices = desired_vertices,\
                                                                                 sigmoid_slope = 100)
        
        '''    
        
        #self.test_polytope_gradient.Gamma_gradient(self.joint,sigmoid_slope=sigmoid_slope_val)
        '''
        Gamma_gradient_all =  hstack((self.test_polytope_model.Gamma_plus_flat,\
                                      self.test_polytope_model.Gamma_minus_flat))
        '''
            
        Gamma_gradient_test = Gamma_min
        Gamma_gradient_hat_test = Gamma_min_softmax
        #Gamma_gradient_hat_test_P = gamma_hat



        error = []
        error2 = []
        kappa_grad = []
        kappa_second_grad = []
        
        analytical_gamma_hat_1_print = []
        
        numerical_gamma_hat_1_print = []
        
        analytical_gamma_hat_2_print = []
        
        numerical_gamma_hat_2_print = []
        
        index_capacity_margin = []
        
        nm_tes=[]
        
        CN_arr = []
        
        kappa_arr = []
        kappa_numerical_arr = []
        
        first_iteration = True
        gamma_gradient_abs = False
        flip_switch = 1.0
        kappa_before_grad = -1000000
        for z in test_domain:
            
            
                
            
            
            if not first_iteration:
                
                Gamma_gradient_test_last = Gamma_gradient_test
                Gamma_gradient_hat_test_last = Gamma_gradient_hat_test
                #Gamma_gradient_hat_test_P_last = Gamma_gradient_hat_test_P
                self.q_joint[self.joint]=z
                #self.q_joint[0:3] = z
                
                #print(' self.joint')
                #print( self.joint)
    
                self.test_robot_model.urdf_transform(self.q_joint)
                h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
                    get_polytope_hyperplane(self.test_robot_model.jacobian_hessian,self.test_robot_model.active_joints,self.cartesian_dof_input,
                                                                      self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
                
                
                Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat = get_capacity_margin(self.test_robot_model.jacobian_hessian,\
                                                            n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,self.test_robot_model.active_joints,\
                                                                self.cartesian_dof_input,self.qdot_min,self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
                
                    
                    
                    
                d_Gamma_all = Gamma_hat_gradient_joint(self.test_robot_model.jacobian_hessian,self.test_robot_model.Hessian,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                        p_minus_hat,self.qdot_min,self.qdot_max,\
                            self.cartesian_desired_vertices,self.joint,self.sigmoid_slope)
                
           

                

                
                Gamma_gradient_test = Gamma_min
                
                Gamma_gradient_hat_test = Gamma_min_softmax
                
                #print('Gamma_gradient_hat_test',Gamma_gradient_hat_test)
                
                #input('stop here')
                
                #Gamma_gradient_hat_test_P = gamma_hat
        
                #print('Gamma_gradient_hat_test_softmax',Gamma_gradient_hat_test)
                
                #print('Gamma_gradient_hat_test_min',self.test_polytope_model.Gamma_min)
                
                #print('Gamma_gradient_hat_test_2',Gamma_gradient_hat_test_2)
                #Gamma_gradient_test_2 = Gamma_plus
                #print('self.test_polytope_model.Gamma_total',self.test_polytope_model.Gamma_total)
                    
                #input('wait')
                numerical_gradient = ((Gamma_gradient_test - Gamma_gradient_test_last) / step)
                
                numerical_gradient_gamma_hat = ((Gamma_gradient_hat_test - Gamma_gradient_hat_test_last) / step)
                
                #numerical_gradient_P = ((Gamma_gradient_hat_test_P - Gamma_gradient_hat_test_P_last) / step)
                print('numerical_gradient',numerical_gradient)
                
                print('numerical_gradient_gamma_hat',numerical_gradient_gamma_hat)
                
                #input('Without OOPS stop first')
                
                #print('numerical_gradient',numerical_gradient)
                
                #print('numerical_gradient_2',numerical_gradient_2)
                #self.test_polytope_gradient.Gamma_gradient(self.joint,100)
                
                
                
                #analytical_gradient = (self.test_polytope_gradient.d_Gamma_hat_d_Gamma[self.test_polytope_model.Gamma_min_index])*(self.test_polytope_gradient.d_Gamma_all[self.test_polytope_model.Gamma_min_index])
                
               
                
                # For LSE
                Gamma_all_array = -1*Gamma_total_hat
                
                print('Gamma_all_array',Gamma_all_array)
                #input('Gamma_all stop - No OOPS')
                
                unravel_index_min = unravel_index(argmax(Gamma_all_array,axis=None),Gamma_all_array.shape)

                print('minimum index',unravel_index_min)
                #input('stop here')
                
                #analytical_gradient = analytical_gradient_sum/(sum(np.exp(self.test_polytope_model.Gamma_total)))
                #LSE_d_gamma = exp_sum(Gamma_all_array[self.test_polytope_model.Gamma_min_index_hat],Gamma_all_array)
                #print('exp_normalize(Gamma_all_array)',
                
                # negative sign in equation 34 as gamma_hat is negative of log
                
                
                #minimum_index_array = argpartition(Gamma_all_array,-3)[-3:]
                
                
                
                ## Working version for single minimum index array
                d_gamma_max_dq_min = -1.0*d_Gamma_all[unravel_index_min]
                
               
                ### New versoin to test
                d_gamma_max_dq = -1.0*d_Gamma_all
                
                print(' d_Gamma_all', d_Gamma_all)
                #input('stop here')
                
                #print('self.test_polytope_gradient.d_Gamma_all[self.test_polytope_model.Gamma_min_index_hat]',self.test_polytope_gradient.d_Gamma_all)
                #### //////////// Got answer similar to Philips code---- DOnt delete this for reference
                #analytical_gradient_tested_with_philip = d_gamma_max_dq*LSE_d_gamma 
                
                
                #print('exp_normalize(Gamma_all_array[self.test_polytope_model.Gamma_min_index_hat])',exp_normalize(Gamma_all_array[self.test_polytope_model.Gamma_min_index_hat]))
                #input('wait here')
                ################################## Dont delete above code - for testing
                
                
                
                
                #if (self.test_polytope_model.Gamma_min_index[0] > (len(self.test_polytope_model.Gamma_total)/2.0)) and (d_gamma_max_dq > 0):
                
                
                

                    
               
                analytical_gradient = 1.0*d_gamma_max_dq_min
                
                #analytical_gradient = flip_switch*self.test_polytope_gradient.d_gamma_max_dq
                
                ## Working version
                #print('exp_normalize(Gamma_all_array)',exp_normalize(100*Gamma_all_array)/1.0)
                #print('Gamma_all_array',Gamma_all_array)
                #analytical_gradient_gamma_hat = 1.0*exp_normalize(Gamma_all_array[self.test_polytope_model.Gamma_min_index_hat])*d_gamma_max_dq_min
                #analytical_gradient_gamma_hat = 1.0*exp_normalize(500*Gamma_all_array)[self.test_polytope_model.Gamma_min_index_hat]*d_gamma_max_dq_min
                #print('100*Gamma_all_array',500*Gamma_all_array)
                #print('exp_normalize',exp_normalize(500*Gamma_all_array))
                #print('exp_normalize',exp_normalize(500*Gamma_all_array)[self.test_polytope_model.Gamma_min_index_hat])
                analytical_gradient_gamma_hat = 1.0*exp_normalize(self.sigmoid_slope*Gamma_all_array)[unravel_index_min]*d_gamma_max_dq[unravel_index_min]
                #input('work here')
                
                
                numerical_gamma_hat_1_print.append(Gamma_gradient_hat_test)
                numerical_gamma_hat_2_print.append(Gamma_gradient_test)
                
                
                analytical_gamma_hat_2_print.append(analytical_gradient_gamma_hat)
                analytical_gamma_hat_1_print.append(analytical_gradient)
                
                
                print('self.sigmoid_slope',self.sigmoid_slope)
                print('analytical_gradient_gamma_hat with OOPS',analytical_gradient_gamma_hat)
                print('analytical_gradient',analytical_gradient)
                
                #input('stop here')
                
                
                '''
                print('numerical_gradient',numerical_gradient)
                
                print('numerical_gradient_gamma_hat is',numerical_gradient_gamma_hat)
                
                #print('numerical_gradient_gamma_hat Philip is',numerical_gradient_P)
                
                
                
                print('analytical_gradient',analytical_gradient)
                print('analytical_gradient_gamma_hat',analytical_gradient_gamma_hat)
                #print('analytical_gradient_Philip',analytical_gradient_P)
                
                print('Gamma minimum is',self.test_polytope_model.Gamma_min)
                

                print('Gamma_gradient_test_last',Gamma_gradient_test_last)
                
                print('Gamma_gradient_test',Gamma_gradient_test)
                
                
                #print('old analytical_gradient',analytical_gradient_old)
                
                print('new analytical_gradient',analytical_gradient)
                print('d_gamma_max_dq',d_gamma_max_dq)
                '''
                

                

                
                

                
                polytope_verts,polytope_faces = velocity_polytope_withfaces(self.test_robot_model.jacobian_hessian[0:3,:],self.qdot_max,self.qdot_min)
                

                #input('www here')
                faces_verts = {}
                for i in range(len(polytope_faces)):
                    xs = [polytope_verts[0,polytope_faces[i,0]],polytope_verts[0,polytope_faces[i,1]],polytope_verts[0,polytope_faces[i,2]]]
                    ys = [polytope_verts[1,polytope_faces[i,0]],polytope_verts[1,polytope_faces[i,1]],polytope_verts[1,polytope_faces[i,2]]]
                    zs = [polytope_verts[2,polytope_faces[i,0]],polytope_verts[2,polytope_faces[i,1]],polytope_verts[2,polytope_faces[i,2]]]
                    verts = list(zip(xs,ys,zs))
                    faces_verts[i] = verts
                    ax_poly.add_collection3d(Poly3DCollection(faces_verts[i],alpha=0.5,edgecolors='k'))
                
                
                #ax_poly.add_collection3d(Poly3DCollection(faces_verts[i]))
                #faces_vertices = polytope_verts[]
                #plot_polytope_faces(faces_verts,ax_poly)
                '''
                capacity_margin_vertex = self.cartesian_desired_vertices[self.test_polytope_model.Gamma_min_index[1]]
                capacity_margin_normal = self.test_polytope_model.n_k[self.test_polytope_model.Gamma_min_index[0]]
                capacity_margin_hyperplane_vertex = self.test_polytope_model.p_plus_hat[self.test_polytope_model.Gamma_min_index[0]]
                if not self.test_polytope_model.Gamma_min_plus:
                    capacity_margin_normal = -1*capacity_margin_normal
                    capacity_margin_hyperplane_vertex = self.test_polytope_model.p_minus_hat[self.test_polytope_model.Gamma_min_index[0]]
                #else:
                
                ax_poly.scatter3D(capacity_margin_vertex[0],capacity_margin_vertex[1],capacity_margin_vertex[2],s=100)
                ax_poly.scatter3D(capacity_margin_hyperplane_vertex[0],capacity_margin_hyperplane_vertex[1],capacity_margin_hyperplane_vertex[2],s = 100)
                capacity_facet = capacity_margin_vertex + capacity_margin_normal*self.test_polytope_model.Gamma_min
                ax_poly.scatter3D(capacity_facet[0],capacity_facet[1],capacity_facet[2],s=100)
                #ax_poly.plot(capacity_margin_vertex,capacity_margin_vertex+(1000*capacity_margin_normal))
                
                ax_poly.scatter3D(polytope_verts[0,polytope_faces[0,0]],polytope_verts[1,polytope_faces[0,0]],polytope_verts[2,polytope_faces[0,0]])
                ax_poly.scatter3D(polytope_verts[0,polytope_faces[0,1]],polytope_verts[1,polytope_faces[0,1]],polytope_verts[2,polytope_faces[0,1]])
                ax_poly.scatter3D(polytope_verts[0,polytope_faces[0,2]],polytope_verts[1,polytope_faces[0,2]],polytope_verts[2,polytope_faces[0,2]])
                #faces_verts = [polytope_verts[polytope_faces[0,0]],polytope_verts[polytope_faces[0,1]],polytope_verts[polytope_faces[0,2]]]
                
                
                
                

                #input('wait here')
                
                plt.pause(0.01)
                ax_poly.cla()
                plt.show()
                '''
                index_capacity_margin.append(unravel_index_min)
                kappa_before_grad = Gamma_gradient_hat_test
                error_gamma_gradient = (norm(numerical_gradient-analytical_gradient))
                
                error.append( error_gamma_gradient )
                print('error gamma gradient is')
                print(error_gamma_gradient)
                
                
                error_gamma_hat_gradient = ((numerical_gradient_gamma_hat-analytical_gradient_gamma_hat)/(numerical_gradient_gamma_hat))*100.0
                print('error gamma gradient_hat is',error_gamma_hat_gradient)
                #print()
                error2.append( error_gamma_hat_gradient)
                
                kappa_grad.append(analytical_gradient_gamma_hat) 
                
                ### CN performance index 
                
                #J_pinv = robot_functions.getJ_pinv(self.test_robot_model.jacobian_hessian,0.9555)
                
                #manip = matmul(self.test_robot_model.jacobian_hessian,transpose(self.test_robot_model.jacobian_hessian))
                
                U,S,V = svd(self.test_robot_model.jacobian_hessian)
                
                S_max = max(S)
                S_min = min(S)
                
                CN = S_min/(S_max*1.0)
                
                
                
                print('CN',CN)
                
                CN_arr.append(CN)
                
                print('kappa - actual is',Gamma_gradient_test)
                kappa_numerical_arr.append(Gamma_gradient_test)
                
                print('kappa - Estimated is',Gamma_gradient_hat_test)
                kappa_arr.append(Gamma_gradient_hat_test)
                
                
 
                
                #print('kappa Second Gradient - Estimated is',self.test_polytope_gradient.d_softmax_dq)
                #kappa_second_grad.append(self.test_polytope_gradient.d_softmax_dq)
                
                
                
                
                
                
                
                print('index contributing to capacity margin is',unravel_index_min)
                #print('index contributing to capacity margin array is',minimum_index_array)
                #print('error gamma gradient_hat Philip code is')
                #error3.append(np.linalg.norm(numerical_gradient_P - analytical_gradient_P))
                #print(np.linalg.norm(numerical_gradient_P-analytical_gradient_P))
                
                #input('wait here')
                
                #print('error',error)
                #print('error2',error2)
                
                #input('wait here')
            first_iteration = False

        plt.figure()
        plt.plot(error[:], 'r')
        plt.xlabel("Steps")
        plt.ylabel("Error (2-norm) Gamma ")
        
        plt.figure()

        plt.plot(error2[:], 'g')
        plt.xlabel("Steps")
        plt.ylabel("Error (%) Kappa ")
        
        plt.figure()

        plt.plot(error2[:], 'k', label='Kappa Gradient')
        plt.legend(loc='best')
        
        plt.xlabel("Steps")
        plt.ylabel("Analytical Gradient - Kappa ")
        
        
        plt.figure()
        plt.plot(CN_arr[:], 'r', label='CN')
        plt.xlabel("Steps")
        plt.ylabel("Condition Number ")
        
        
        
        plt.figure()
        plt.plot(kappa_arr[:], 'cyan', label='Kappa Estimated')
        plt.plot(kappa_numerical_arr[:], 'r', label='Kappa Actual')
        plt.legend(loc='best')
        plt.xlabel("Steps")
        plt.ylabel("Capacity Margin - Kappa ")
        
        
        '''
                
        plt.figure()
        plt.plot(kappa_second_grad[:], 'y')
        plt.legend(loc='best')
        plt.xlabel("Steps")
        plt.ylabel("Kappa Second order Gradient")
        '''
        
        plt.figure()
        plt.plot(index_capacity_margin[:], 'y', label='Vertex index')
        plt.legend(loc='best')
        plt.xlabel("Steps")
        plt.ylabel("Vertex contributing to Capacity Margin")
        
        #index_capacity_margin
        #plt.plot(error3[:], 'b')
        #plt.plot(analytical_gamma_hat_1_print[:],'k')
        #plt.plot(analytical_gamma_hat_2_print[:],'cyan')
        
        #plt.plot(numerical_gamma_hat_1_print[:],'k')
        #plt.plot(numerical_gamma_hat_2_print[:],'cyan')
        
        #plt.legend(["", "orange"], loc=0, frameon=legend_drawn_flag)
       
        plt.show()        
        #max(error)


    def test_h_vs_h_hat(self,limits,step):
        
        from numpy.random import randn
        from numpy.linalg import norm
        from linearalgebra import V_unit
        from copy import deepcopy
        from numpy import amax,max,exp,dot,matmul,ones
        from polytope_functions import get_hyperplane_parameters,get_gamma,get_gamma_hat
        from robot_functions import exp_sum,exp_normalize,smooth_max_gradient
        from numpy import unravel_index,argmax,min

        test_domain = arange(limits[0], limits[1], step)
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        print('self.joint',self.joint)
        self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,self.cartesian_dof_input,
                                                              self.qdot_min,self.qdot_max,self.cartesian_desired_vertices)
        
        ## Get Gamma and Gamma_hat
        #Gamma = self.test_polytope_model.Gamma_min
        #Gamma_hat = self.test_polytope_model.Gamma_min_softmax
        
        
        hplus_error = []
        hminus_error = []
        hplus_max_array = []
        hplus_min_array = []
        
        hplus_hat_max_array = []
        hplus_hat_min_array = []
        
        hminus_max_array = []
        hminus_min_array = []
        
        
        hminus_hat_max_array = []
        hminus_hat_min_array = []

        first_iteration = True
        for z in test_domain:
            
            if not first_iteration:
                #Gamma_last = Gamma
                #Gamma_hat_last = Gamma_hat
                self.q_joint[self.joint]=z
                
                self.test_robot_model.urdf_transform(self.q_joint)
                self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,self.cartesian_dof_input,
                                                                  self.qdot_min,self.qdot_max,self.cartesian_desired_vertices)
                #self.test_polytope_model.plot_polytope(canvas_input,True,True)
                
                        ## Get h and h_hat
                hplus = self.test_polytope_model.h_plus
                
                hplus_max_array.append(max(hplus))
                hplus_min_array.append(min(hplus))
                
                
                
                hminus = self.test_polytope_model.h_minus
                
                hminus_max_array.append(max(hminus))
                hminus_min_array.append(min(hminus))
                
                hplus_hat = self.test_polytope_model.h_plus_hat
                
                
                hplus_hat_max_array.append(max(hplus_hat))
                hplus_hat_min_array.append(min(hplus_hat))
                
                hminus_hat = self.test_polytope_model.h_minus_hat
                
                hminus_hat_max_array.append(max(hminus_hat))
                hminus_hat_min_array.append(min(hminus_hat))
                
                
                hplus_error.append(norm(hplus-hplus_hat))
                hminus_error.append(norm(hminus-hminus_hat))
                #Gamma_array.append(Gamma)
                
                #Gamma_hat_array.append(Gamma_hat)
            first_iteration = False
        
        
        plt.plot(hplus_error[:], 'r',label = "hplus vs hplus hat hat Error")
        
        plt.plot(hminus_error[:], 'k',label = "hminus vs hminus hat hat Error")
        
        
        plt.legend(loc="upper right")
        plt.xlabel("Steps")
        plt.ylabel("Error")
                
        plt.figure()
        
        plt.plot(hplus_max_array[:], 'r',label = "hplus - Max")
        plt.plot(hplus_min_array[:], 'g',label = "hplus -  Min")
        
        
        plt.plot(hplus_hat_max_array[:], 'k',label = "hplus hat - Max")
        plt.plot(hplus_hat_min_array[:], 'cyan',label = "hplus hat-  Min")
        
        plt.legend(loc="upper right")
        plt.xlabel("Steps")
        plt.ylabel("Value")
        
        
        plt.figure()
        
        
        plt.plot(hminus_max_array[:], 'r',label = "hminus - Max")
        plt.plot(hminus_min_array[:], 'g',label = "hminus -  Min")
        
        
        plt.plot(hminus_hat_max_array[:], 'k',label = "hminus hat - Max")
        plt.plot(hminus_hat_min_array[:], 'cyan',label = "hminus hat-  Min")
        
        

        plt.legend(loc="upper right")
        plt.xlabel("Steps")
        plt.ylabel("Value")


    def test_Gamma_vs_Gamma_hat(self,limits,step):
        
        from numpy.random import randn
        from numpy.linalg import norm
        from linearalgebra import V_unit
        from copy import deepcopy
        from numpy import amax,max,exp,dot,matmul,ones
        from polytope_functions import get_hyperplane_parameters,get_gamma,get_gamma_hat
        from robot_functions import exp_sum,exp_normalize,smooth_max_gradient
        from numpy import unravel_index,argmax,min

        test_domain = arange(limits[0], limits[1], step)
        
        self.q_joint[self.joint] = limits[0]
        self.test_robot_model.urdf_transform(self.q_joint)
        
        print('self.joint',self.joint)
        self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,self.cartesian_dof_input,
                                                              self.qdot_min,self.qdot_max,self.cartesian_desired_vertices)
        
        ## Get Gamma and Gamma_hat
        #Gamma = self.test_polytope_model.Gamma_min
        #Gamma_hat = self.test_polytope_model.Gamma_min_softmax
        
        
        gamma_error = []
        Gamma_array = []
        Gamma_hat_array = []
        first_iteration = True
        for z in test_domain:
            
            if not first_iteration:
                #Gamma_last = Gamma
                #Gamma_hat_last = Gamma_hat
                self.q_joint[self.joint]=z
                
                self.test_robot_model.urdf_transform(self.q_joint)
                self.test_polytope_model.generate_polytope_hyperplane(self.test_robot_model,self.cartesian_dof_input,
                                                                  self.qdot_min,self.qdot_max,self.cartesian_desired_vertices)
                #self.test_polytope_model.plot_polytope(canvas_input,True,True)
                
                        ## Get Gamma and Gamma_hat
                Gamma = self.test_polytope_model.Gamma_min
                Gamma_hat = self.test_polytope_model.Gamma_min_softmax
                gamma_error.append(norm(Gamma-Gamma_hat))
                Gamma_array.append(Gamma)
                
                Gamma_hat_array.append(Gamma_hat)
            first_iteration = False
        
        plt.plot(gamma_error[:], 'r',label = "Gamma vs Gamma hat Error")
        plt.legend(loc="upper right")
        plt.xlabel("Steps")
        plt.ylabel("Error")
                
        plt.figure()
        
        plt.plot(Gamma_array[:], 'cyan',label = "Gamma")
        plt.plot(Gamma_hat_array[:], 'b',label = "Gamma hat")
        plt.legend(loc="upper right")
        plt.xlabel("Steps")
        plt.ylabel("Value")
    
    
    
        

if __name__ == '__main__':
    
    global sigmoid_slope_val
    sigmoid_slope_val = 250
    
    
    sys.path.insert(0,'C:/Users/keerthi.sagar/Dropbox/Github_repos/pygradientpolytope/pygradientpolytope/pygradientcapacity/')
    #import mat
    '''
    canvas_input = GenerateCanvas()
    canvas_input.set_params(view_ang1=30,view_ang2=45,x_limits=[-2,2],y_limits=[-2,2],z_limits=[-2,2],axis_off_on = True)
    canvas_input.generate_axis()
    '''
    
    canvas_polytope = GenerateCanvas()
    canvas_polytope.set_params(view_ang1=30,view_ang2=45,x_limits=[-0.8,0.8],y_limits=[-0.8,0.8],z_limits=[-0.8,0.8],axis_off_on = True)
    canvas_polytope.generate_axis()
    
    test_gradient = TestGradient()
    
    ### Launch robot here and also generate random initial joint angles here
    '''
    test_gradient.launch_robot(robot_name = 'kr4r600',urdf_address="URDF/kuka_kr4_support/urdf/",
                             end_effector_offset=0.08,canvas_plot= None,
                             end_effector_position_offset=[0.0001,0.0001,0.0001],
                             plot_primitives_on=False,plot_screws_on=False,plot_urdf_on = False,world_frame_input='base')
    '''
    
    sys.path.insert(0,'C:/Users/keerthi.sagar/Dropbox/Github_repos/stormkinpy/stormkinpy/')
    test_gradient.launch_robot(robot_name = 'ur5',urdf_address="URDF/UR5/",
                             end_effector_offset=0.003,canvas_plot= None,
                             end_effector_position_offset=[0.0001,0.0001,0.0001],
                             plot_primitives_on=False,plot_screws_on=False,plot_urdf_on=False,world_frame_input='world')
    
    
    
    desired_vertices = array([[0.20000, 0.50000, 0.50000],
                             [0.50000, -0.10000, 0.50000],
                             [0.50000, 0.50000, -0.60000],
                             [0.50000, -0.10000, -0.60000],
                             [-0.30000, 0.50000, 0.50000],
                             [-0.30000, -0.10000, 0.50000],
                             [-0.30000, 0.50000, -0.60000],                         
                             [-0.30000, -0.10000, -0.60000]])
    '''
    
    desired_vertices = array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 0, 1],
                         [0, 1,1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [1, 1, 0],                         
                         [0, 1, 0]])
    '''
    desired_vertices = 1*desired_vertices*0.3
    qdot_min_input = array([-0.3,-0.3,-0.3,-0.3,-0.3,-0.3])*(-10.0)
    qdot_max_input = array([0.6,0.6,0.6,0.6,0.6,0.6])*10.0
    test_gradient.polytope_parameters(cartesian_dof_input = array([True,True,True,False,False,False]), qdot_min = qdot_min_input, qdot_max = qdot_max_input, cartesian_desired_vertices=desired_vertices,sigmoid_slope=sigmoid_slope_val)
    
    
    
    
    #test_gradient.compute_polytope_gradient()
    #test_gradient.test_polytope_gradient.jacobian_gradient(test_gradient.joint)
    #test_gradient.test_polytope_gradient.hyperplane_gradient(test_gradient.joint,sigmoid_slope = sigmoid_slope_val)
    #test_gradient.test_polytope_gradient.Gamma_gradient(test_gradient.joint,sigmoid_slope_val)
    
    #test_gradient.test_polytope_model.plot_polytope(canvas_polytope,True,False)
    
    #plt.pause(0.01)
    
    
    #limits = [0.0,pi/2.0]
    #step = 0.05
    
    

    # test_plot()
    # print("Max error sigmoid gradient",test_sigmoid_gradient([-5.0,5.0],0.001,10))
    # print("Max error of cross product gradient",test_cross_product_gradient([-1.0, 1.0], 0.0001))
    #print("Max error of norm vector gradient",test_vector_norm_gradient([-1.0, 1.0], 0.0001))
    #print("Max error of normalized cross product gradient", test_normalized_cross_product_gradient([-1.0, 1.0], 0.0001))
    #print("Max error of Hessian", test_hessian([0.0, np.pi], 0.01))
    #print("Max error of Hessian", test_hessian([0.0, 0.03], 0.01))
    #print("Max error of smooth max", test_smooth_max_gradient([-5.0, 5.0],0.001))
    # print("Max error of smooth max", test_smooth_min_gradient([-5.0, 5.0],0.0001))
    #print("Max error of test_hyperplanes ", test_hyperplanes([-1.0, 1.0],0.01))
    #print("Comparison Gamma versus Gamma hat ",test_gamma_versus_gammahat([-10.0, 10.0],0.1))
    #print("Testing gamma hat gradient", test_gamma_hat_gradient([-1.0, 0.0], 0.01))
    
    # Tested- Working
    #print("Testing hessian here", test_gradient.test_hessian(limits = [0.0,pi/3.0], step = 0.01))
    
    # Tested - Working
    #print("Testing twist here", test_gradient.test_twist_gradient(limits = [0.0,pi/3.0], step = 0.01))
    
    # Tested - Working
    #print("Testing cross product here", test_gradient.test_cross_product_norm_gradient(limits = [0.0,pi/3.0], step = 0.01))
    
    
    # Tested - Working
    #print("Testing normal gradient here", test_gradient.test_normal_gradient(limits = [0.3,pi/3.0], step = 0.01))
    
    # Tested- Working- But error isnt very less
    #print("Testing sigmoid gradient here", test_gradient.test_sigmoid_gradient(limits = [0.3,pi/3.0], step = 0.001))
    
    print("Testing hyperplane gradient here",test_gradient.test_hyperplane_gradient(limits = [pi/3.0,pi/2.0], step = 0.001))
    
    #print("Testing test_gamma_gradient here",test_gradient.test_gamma_gradient(limits = [0.0,pi/3.0], step = 0.01))
    
    
    #print("Testing test_gamma_hat_gradient here",test_gradient.test_gamma_hat_gradient(limits = [pi/3.0,pi/2.0], step = 0.01))

    #print("Testing gamma vs gamma hat error", test_gradient.test_Gamma_vs_Gamma_hat([-1.0, 1.0], 0.001))
    
    #print("Testing h vs h hat error", test_gradient.test_h_vs_h_hat([-1.0, 1.0], 0.01))
    #print("Testing gamma gradient", test_gammas_gradient([-1.0, 0.0], 0.001))

    # Something not right with the gamma gradient and I can't figure it out yet
    # Need to verify that gamma gradient in python is same as result from octave

    # Applications
