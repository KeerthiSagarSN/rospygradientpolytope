# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:56:14 2022

@author: keerthi.sagar
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:50:56 2022

@author: keerthi.sagar
"""
from numpy import empty,shape,cross,zeros,dot,transpose,matmul
from numpy.linalg import norm
import time


from numpy import empty,shape,cross,zeros,dot,transpose,matmul,array
from numpy.linalg import norm
from rospygradientpolytope.linearalgebra import V_unit, check_ndarray
from rospygradientpolytope.robot_functions import exp_sum,exp_normalize,smooth_max_gradient,sigmoid
from math import isnan
from rospygradientpolytope.gradient_functions import normal_gradient, normal_twist_projected_gradient, sigmoid_gradient

    

## Eq.42 is here
## All values here are with respect the estimated parameters in the polytope

def hyperplane_gradient(JE,H,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                        p_minus_hat,qdot_min,qdot_max,\
                            test_joint,sigmoid_slope):
    
    
    
    
    

    
    #from polytope_gradient_functions import normal_hyperplane_gradient,sigmoid_gradient,sigmoid
    # Eq. 42 here Testing dh_plus_dq here 
    #Nmatrix = self.polytope_model.Nmatrix
    #Nnot = self.polytope_model.Nnot
    h_plus = h_plus_hat
    h_minus = h_minus_hat
    J_Hessian = JE
    n = n_k
    
    deltaqq = qdot_max - qdot_min
    #print('test_joint',test_joint)
    #print('deltaqq without oops',deltaqq)
    # Cycling through all the noramls of the hyper plane here:
    #n = empty(shape=(len(self.polytope_model.h_plus),3))
    
    dn_dq = empty(shape = (len(h_plus),3))
    
   
    
    n_T_vk = empty(shape = (len(Nmatrix), shape(Nnot)[1]) )
    vk = {}
    
    vk_dq = {}
    h_plus_gradient = zeros(shape = (len(h_plus)))
    
    
    
    h_minus_gradient = zeros(shape = (len(h_plus)))
    
    ## This function below is the bottleneck - Need to rewrite in a manner where it is not in a for-loop and complete matrix format is followed for dh/dq



    for normal_index in range(len(Nmatrix)):
        
        
        # Get the corresponding twist indices
        twist_index_1 = Nmatrix[normal_index, 0]
        twist_index_2 = Nmatrix[normal_index, 1]
        # Normals based on the screw vectors 
        
        dn_dq[normal_index,:] = normal_gradient(twist_index_1, twist_index_2,JE,H, test_joint)
        
        for index_projected_screw in range(len(Nnot[normal_index])):
            

            twist_index_projected = Nnot[normal_index,index_projected_screw]
            # All screws which do not generate the normal are projected on the normal: 
            vk = J_Hessian[0:3,twist_index_projected]
            
            
            
            #vk_dq[normal_index,index_projected_screw] = self.twist_gradient(twist_index_projected, test_joint)
            
            
            #vk_dq[normal_index,index_projected_screw] = H[0:3,Nnot[normal_index,index_projected_screw],joint_number]          
            
            #n_T_vk[normal_index,index_projected_screw] = dot(transpose(n[normal_index]),vk)
            
            nT_dot_vk = dot(transpose(n[normal_index]),vk)
            # Eq. 42 here
            
            #d_nT_vk_dq = dot(dn_dq[normal_index],vk[normal_index,index_projected_screw]) + dot(n_vk[normal_index],vk_dq[normal_index,index_projected_screw])
            ## Right side termn of 42 
            
            d_nT_vk_dq = normal_twist_projected_gradient(twist_index_1,twist_index_2,twist_index_projected,JE,H,test_joint)
            
            #deltaq_k = 
            
            sig_nT_vk = sigmoid(nT_dot_vk,sigmoid_slope)
            
            sig_negative_nT_vk = sigmoid(nT_dot_vk,-1.0*sigmoid_slope)
            
            #d_sig_nT_vk_dq = dot(sigmoid_gradient(n_T_vk[normal_index,index_projected_screw],100),d_nT_vk_dq)
            
            d_sig_nT_vk_dq = sigmoid_gradient(twist_index_1, twist_index_2, twist_index_projected,JE,H,test_joint,sigmoid_slope)
            d_negative_sig_nT_vk_dq = sigmoid_gradient(twist_index_1, twist_index_2, twist_index_projected,JE,H,test_joint, -sigmoid_slope)
            
            # Initialize the parameter
            dh_plus_dq = 0
            
            dh_plus_dq =  d_sig_nT_vk_dq*deltaqq[Nnot[normal_index,index_projected_screw]]*n_T_vk[normal_index,index_projected_screw] + \
                                                                            (sig_nT_vk*deltaqq[Nnot[normal_index,index_projected_screw]]*d_nT_vk_dq)
            
            # Initialize dh_minus parameter                
            
            dh_minus_dq = 0                
            
            dh_minus_dq =  d_negative_sig_nT_vk_dq*deltaqq[Nnot[normal_index,index_projected_screw]]*n_T_vk[normal_index,index_projected_screw] + \
                                                                            (sig_negative_nT_vk*deltaqq[Nnot[normal_index,index_projected_screw]]*d_nT_vk_dq)
            
            #print('dh_plus_dq',dh_plus_dq)
            
            h_plus_gradient[normal_index] +=  dh_plus_dq
            
            h_minus_gradient[normal_index] += dh_minus_dq
            
        
    

      
            
    d_h_plus_dq = h_plus_gradient
    d_h_minus_dq = h_minus_gradient
    
    #print('d_h_plus_dq',d_h_plus_dq)
    
    ##print('d_h_minus_dq',d_h_minus_dq)
    #print('dn_dq',dn_dq)
    #input('inside hyperplane gradient - no oops')
    #self.dn_dq = dn_dq
    
    return d_h_plus_dq, d_h_minus_dq, dn_dq

#############################################################
## All values here are with respect the estimated parameters in the polytope
 
    
def Gamma_hat_gradient_joint(JE,H,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                        p_minus_hat,qdot_min,qdot_max,\
                            cartesian_desired_vertices,test_joint,sigmoid_slope_joint):
    
    from numpy import empty,shape,cross,zeros,dot,transpose,matmul,array,hstack,argwhere,argmin, unravel_index
    from numpy.linalg import norm
    from rospygradientpolytope.linearalgebra import V_unit, check_ndarray
    from rospygradientpolytope.robot_functions import exp_normalize
    
    from rospygradientpolytope.polytope_gradient_functions import hyperplane_gradient
    # Eq. 42 here Testing dh_plus_dq here 
    h_plus = h_plus_hat
    h_minus = h_minus_hat
    J_Hessian = JE
    n = n_k
    
    v_k_d = cartesian_desired_vertices
    
    
    deltaqq = qdot_max - qdot_min
    deltaqq = transpose(array([deltaqq]))

    deltaqmin = transpose(array([qdot_min]))
    
    ## Get all intrinsic parameters of the robot here: 
    
    d_h_plus_dq, d_h_minus_dq, dn_dq = hyperplane_gradient(JE,H,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                p_minus_hat,qdot_min,qdot_max,\
                    test_joint,sigmoid_slope_joint)
    
    
    ## Get Capacity Margin parameters for the desired polytope
    
    # Cycling through all the noramls of the hyper plane here:
    #n = empty(shape=(len(self.polytope_model.h_plus),3))
    
    # Eq. 36 calculated here - hyperplane parameters are computed here
    
    #self.hyperplane_gradient(test_joint,sigmoid_slope)
    
    d_Gamma_plus = zeros([len(n),len(v_k_d)])
    
    d_Gamma_minus = zeros([len(n),len(v_k_d)])
    

    
    
    #
    
    Gamma_plus_LHS = transpose(array([d_h_plus_dq])) + matmul(matmul(dn_dq,J_Hessian[0:3,:]),deltaqmin) + matmul(matmul(n,H[0:3,:,test_joint]),deltaqmin )
    
    Gamma_minus_LHS = transpose(array([d_h_minus_dq])) + matmul(matmul(dn_dq,J_Hessian[0:3,:]),deltaqmin) + matmul(matmul(n,H[0:3,:,test_joint]),deltaqmin )
    
    #print('v_k_d is',v_k_d)
    
    #input('wait inside gradient function')
    
    for vertex in range(len(v_k_d)):
        #print 'vertex'
        #print vertex
        #print('v_k_d[vertex,:]',v_k_d[vertex,:])
        #print('len(v_k_d[vertex,:])',len(v_k_d[vertex,:]))
        
        
        #print('d_Gamma_plus',Gamma_plus_LHS - matmul(self.dn_dq,transpose(array([v_k_d[vertex,:]]))))
        
        
        d_Gamma_plus[:,vertex] =  transpose(Gamma_plus_LHS - matmul(dn_dq,transpose(array([v_k_d[vertex,:]]))))
       
        d_Gamma_minus[:,vertex] = transpose(Gamma_minus_LHS - matmul(dn_dq,transpose(array([v_k_d[vertex,:]]))))
    
        
    
    ## Eq.30 is here
    
    d_Gamma_plus_flat = check_ndarray(d_Gamma_plus)
    d_Gamma_minus_flat = check_ndarray(d_Gamma_minus)
    
    d_Gamma_all = hstack((d_Gamma_plus_flat,-d_Gamma_minus_flat))
    
    #Eq. 35 is here
    #d_Gamma_hat_d_Gamma = exp_normalize(-Gamma_total)
    #print('d_Gamma_all',d_Gamma_all)

    #print('d_Gamma_plus',unravel_index(d_Gamma_plus.argmin(), d_Gamma_plus.shape))
    
    #input('stop here')
    return d_Gamma_all
    #return self.d_Gamma_plus_flat    
    
    
    ## All values here are with respect the estimated parameters in the polytope
'''
def Gamma_hat_gradient_joint(self,test_joint,sigmoid_slope):
    from numpy import empty,shape,cross,zeros,dot,transpose,matmul,array
    from numpy.linalg import norm
    from linearalgebra import V_unit, check_ndarray
    from robot_functions import exp_sum,exp_normalize,smooth_max_gradient
    
    Nmatrix = self.polytope_model.Nmatrix
    Nnot = self.polytope_model.Nnot
    h_plus = self.polytope_model.h_plus_hat
    h_minus = self.polytope_model.h_minus_hat
    J_Hessian = self.robot_polytope.jacobian_hessian[0:3,:]
    n = self.polytope_model.n_k
    v_k_d = self.polytope_model.v_k_d
    H = self.robot_polytope.Hessian
    
    deltaqq = self.polytope_model.deltaqq
    deltaqq = transpose(array([deltaqq]))
    
    
    self.hyperplane_gradient(test_joint,sigmoid_slope)
    self.Gamma_gradient(test_joint,sigmoid_slope)
    
    ### Analytical gradient of gamma is here: 
        
    
    
    self.d_gamma_max_dq = -1.0*self.d_Gamma_all[self.polytope_model.Gamma_min_index_hat]
    
    
    Gamma_all_array = -1*self.polytope_model.Gamma_total_hat
    
    self.d_LSE_dq = exp_normalize(Gamma_all_array[self.polytope_model.Gamma_min_index_hat])
    
    

    self.d_gamma_hat = 1.0*self.d_LSE_dq*self.d_gamma_max_dq
    
    self.d_softmax_dq = self.d_gamma_hat*(1-self.d_gamma_hat)
    
    print('self.d_softmax_dq',self.d_softmax_dq)
    
    #input('stpo')]

    # Second derivative with respect to itself dsoftmax_i_di



def Gamma_hat_gradient_joint(self,test_joint,sigmoid_slope):
    from numpy import empty,shape,cross,zeros,dot,transpose,matmul,array
    from numpy.linalg import norm
    from linearalgebra import V_unit, check_ndarray
    from robot_functions import exp_sum,exp_normalize,smooth_max_gradient
    
    Nmatrix = self.polytope_model.Nmatrix
    Nnot = self.polytope_model.Nnot
    h_plus = self.polytope_model.h_plus_hat
    h_minus = self.polytope_model.h_minus_hat
    J_Hessian = self.robot_polytope.jacobian_hessian[0:3,:]
    n = self.polytope_model.n_k
    v_k_d = self.polytope_model.v_k_d
    H = self.robot_polytope.Hessian
    
    deltaqq = self.polytope_model.deltaqq
    deltaqq = transpose(array([deltaqq]))
    
    
    self.hyperplane_gradient(test_joint,sigmoid_slope)
    self.Gamma_gradient(test_joint,sigmoid_slope)
    
    ### Analytical gradient of gamma is here: 
        
    
    
    self.d_gamma_max_dq = -1.0*self.d_Gamma_all[self.polytope_model.Gamma_min_index_hat]
    
    
    Gamma_all_array = -1*self.polytope_model.Gamma_total_hat
    
    self.d_LSE_dq = exp_normalize(Gamma_all_array[self.polytope_model.Gamma_min_index_hat])
    
    

    self.d_gamma_hat_dq = 1.0*self.d_LSE_dq*self.d_gamma_max_dq
    
    self.d_softmax_dq = self.d_gamma_hat_dq*(1-self.d_gamma_hat_dq)
    
    #print('self.d_softmax_dq',self.d_softmax_dq)
    
    #input('stpo')

    
    # Second derivative with respect to itself dsoftmax_i_di
'''

def Gamma_hat_gradient(JE,H,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                        p_minus_hat,Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat,\
                        qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope):
    
    

    
    h_plus = h_plus_hat
    h_minus = h_minus_hat
    J_Hessian = JE
    n = n_k
    
    
    
    
    deltaqq = qdot_max - qdot_min
    deltaqq = transpose(array([deltaqq]))
    
    
    ## Number of joints - dq
    d_gamma_hat = zeros(shape(JE)[1])
    d_softmax_dq = zeros(shape(JE)[1])
    
    ## Get all hyperplane parameters
    
    sigmoid_slope_joint = sigmoid_slope
    
    #print('Jacobian inside Gamma_hat_gradient',JE)
    #input('Jacobian inside Gamma_hat_gradient')
    
    for test_joint in range(0,shape(JE)[1]):
        

            
        d_Gamma_all = Gamma_hat_gradient_joint(JE,H,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                        p_minus_hat,qdot_min,qdot_max,cartesian_desired_vertices,test_joint,sigmoid_slope_joint)    
            
        
        #print('d_Gamma_all',d_Gamma_all)
        #Gamma_gradient(test_joint,sigmoid_slope)
        
        ### Analytical gradient of gamma is here: 
            
        
        
        d_gamma_max_dq = -1.0*d_Gamma_all[Gamma_min_index_hat]
        
        #print('Gamma_hat--> check if positive',self.polytope_model.Gamma_total_hat)
        #input('wait here 1')
        #Gamma_all_array = -1*Gamma_total_hat

        #print('d_gamma_max_dq',d_gamma_max_dq)
        Gamma_all_array = -1.0*Gamma_total_hat
        
        ### This was the parameter for the smooth gradient for the robot
        d_LSE_dq_arr = exp_normalize(100.0*Gamma_all_array)

        ### This is the parameter for the UR5 tests- running it again for Sawyer to see the convergence
        #d_LSE_dq_arr = exp_normalize(10000000.0**Gamma_all_array)

        #print('d_LSE_dq_arr',d_LSE_dq_arr)
        #input('stop to test here')
        #d_LSE_dq = max(d_LSE_dq_arr)
        #d_LSE_dq = max(d_LSE_dq_arr)

        d_LSE_dq = d_LSE_dq_arr[Gamma_min_index_hat]



        #d_LSE_dq_min = d_LSE_dq_arr[Gamma_min_index_hat]

        #print('d_LSE_dq',d_LSE_dq_arr)
        #print('test_joint',test_joint)
        #print('d_gamma_max_dq',d_gamma_max_dq)
        #print('d_LSE_dq_min',d_LSE_dq_min)

        
        

        d_gamma_hat[test_joint] = 1.0*d_LSE_dq*d_gamma_max_dq

        #d_gamma_hat[test_joint] = 1.0*d_gamma_max_dq

        #print('d_gamma_hat[test_joint]',d_gamma_hat[test_joint] )
        #print('d_LSE_dq',d_LSE_dq)

        #print('d_gamma_max_dq',d_gamma_max_dq)
        #input('stop here')
        
        #d_softmax_dq[test_joint] = d_gamma_hat[test_joint]*(1-d_gamma_hat[test_joint])
        
        #print('self.d_softmax_dq',self.d_softmax_dq)
        
        #input('stpo')
        if isnan(d_gamma_hat[test_joint]):
            print('self.d_gamma_hat[test_joint]',d_gamma_hat[test_joint])
            print('self.d_LSE_dq',d_LSE_dq)
            print('self.d_gamma_max_dq',d_gamma_max_dq)
            print('self.d_Gamma_all',d_Gamma_all)
            print('self.polytope_model.Gamma_min_index_hat',Gamma_min_index_hat)
            print('self.d_Gamma_all[self.polytope_model.Gamma_min_index_hat]',d_Gamma_all[Gamma_min_index_hat])
            input('stp')
        '''
        if (self.d_softmax_dq > 0) and (self.polytope_model.Gamma_min_index_hat[0] > (len(self.polytope_model.Gamma_total_hat)/2.0)):
            #input('stpo')
            self.d_gamma_hat[test_joint] = 1.0*self.d_gamma_hat[test_joint]
        
        if (self.d_softmax_dq > 0) and (self.polytope_model.Gamma_min_index_hat[0] < (len(self.polytope_model.Gamma_total_hat)/2.0)):
            #input('stpo')
            self.d_gamma_hat[test_joint] = 1.0*self.d_gamma_hat[test_joint]
        '''
    #print('d_gamma_hat',d_gamma_hat)
    return d_gamma_hat
            
    
    # Second derivative with respect to itself dsoftmax_i_di

    
'''
def h_plus_gradient(self):
    self.d_h_plus_dq = 

def h_minus_gradient(self):
    self.d_h_minus_dq = 

def Gamma_min_gradient(self):
'''  
    
        
    
    
        

        