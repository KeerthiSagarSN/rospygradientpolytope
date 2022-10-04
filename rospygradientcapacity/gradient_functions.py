# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:56:56 2022

@author: keerthi.sagar
"""

from numpy import cross,matmul
from linearalgebra import skew
from numpy import cross,transpose,matmul,sqrt

from numpy import matmul,transpose,cross,dot
from linearalgebra import V_unit
from numpy.linalg import norm
from numpy import cross,matmul
    
# Eq.45 is here
# Input Hessian , dq
def jacobian_gradient(H,test_joint):
     
    d_Je_dq = H[:,:,test_joint]
    return d_Je_dq
 


 ## eq.44 is here -- > dv_k_dq
 
def twist_gradient(twist_index,H,test_joint):
   
    
    '''    
    Args
    Twist index
    test_joint = dq
    Hessian - H
    
    
    Function for 3- DOF - Linear velocity
    
    '''
    return H[0:3,twist_index,test_joint]
         
 
# Eq. 37 and 38 is here
def cross_product_gradient(twist_index_1,twist_index_2,JE,H,test_joint): # Input 
    

    '''
    ### d(v1xv2)/dq = (dv1/dq)xv2 + v1x(dv2/dq)
    ## We are now concerned only with linear velocity, so we leave out angular velocity thus
    ## the slicing of [0:3]
    
    Args
    Twist index :  v1
    Twist index 2: v2
    test_joint: dq
    JE - Jacobian
    H - Hessian
    
    Returns
    
    -------
    d(v1xv2)/dq
    
    -------
    
    '''
    

    dv1_dq = twist_gradient(twist_index_1,H,test_joint)
    dv2_dq = twist_gradient(twist_index_2,H,test_joint)
    
    v1 = JE[0:3,twist_index_1]
    v2 = JE[0:3,twist_index_2]
    
    #return matmul(-skew(v2), dv1_dq) + matmul(skew(v1), dv2_dq)
    
    
    return cross(dv1_dq,v2) + cross(v1,dv2_dq)
 
## Eq.40 is here

def cross_product_norm_gradient(twist_index_1,twist_index_2,JE,H,test_joint):
    
    '''
    
    Partial derivative of a norm of 2 vector
    d(|| v1 x v2 ||)/dq = (((d(v1 x v2)/dq)^T(v1xv2))/(((v1 x v2)^T(v1 x v2))^0.5)
    
    
    Args
    Twist index :  v1
    Twist index 2: v2
    test_joint: dq
    JE - Jacobian
    H - Hessian
    
    Returns
    
    -------
    d(v1xv2)/dq/(||v1xv2||)
    
    -------
    
    '''
    
    # Partial derivative of a norm of 2 vector
    
    v1_x_v2 = cross(JE[0:3,twist_index_1],
                    JE[0:3,twist_index_2])
    numerator = matmul(transpose(cross_product_gradient(twist_index_1, twist_index_2,JE,H, test_joint)),v1_x_v2)
    denom = sqrt(matmul(transpose(v1_x_v2),v1_x_v2))
    
    return numerator*(denom**(-1))
     
     
 
     


## Eq.44 full equation is here --> d_nT_vk_dq
## Function very specific to the paper - Not a generic function

def normal_twist_projected_gradient(twist_index_1,twist_index_2,twist_index_projected, JE,H,test_joint):
    

    
    
    '''
    
    Partial derivative of a norm of 2 vector
    (dnT*vk)/dq = d_nT_dq*v_k + n_T*d_vk_dq
    n_T = (v1xv2)/(||v1xv2||)
    
    Args
    Twist index :  v1
    Twist index 2: v2
    Projected twist: vk
    test_joint: dq
    JE - Jacobian
    H - Hessian
    
    Returns
    
    -------
    dx_dq = (dnT*vk)/dq = d_nT_dq*v_k + n_T*d_vk_dq
    
    -------
    
    '''
    
    v1 = JE[0:3,twist_index_1]
    v2 = JE[0:3,twist_index_2]
    
    vk = JE[0:3,twist_index_projected]
    
    nt = transpose(V_unit(cross(v1,v2)))
    
   
    
    
    
    d_nt_dq = normal_gradient(twist_index_1,twist_index_2,JE,H,test_joint)
    
    d_vk_dq = twist_gradient(twist_index_projected,H,test_joint)
    
    ## Eq. 44 is here
    
    #print('d_nt_dq',d_nt_dq)
    #print('vk',vk)
    #print('n',n)
    #print('d_vk_dq',d_vk_dq)
    
    
    
    dx_dq = matmul(d_nt_dq,vk) + matmul(nt,d_vk_dq)
    
    return dx_dq 
 
 
 # Eq. 41 is here
 
def normal_gradient(twist_index_1,twist_index_2,JE,H,test_joint):
    
          
    '''
    
    Partial derivative of a norm vctor
   dn/dq = d(v1xv2)/dq||v1xv2|| - (v1xv2)d(||v1xv2||)/dq
           ---------------------------------------------
                          ||v1xv2||^2
    
    Args
    Twist index :  v1
    Twist index 2: v2
    test_joint: dq
    JE - Jacobian
    H - Hessian
    
    Returns
    
    -------
    dn_dq
    
    -------
    
    '''
        
    #  (d(v1xv2)/dq)( ||v1 x v2||)  -  (v1 x v2)(d(|| v1 x v2 ||)/dq)/(|| v1 x v2 ||^2)
    

    
    v1_x_v2 = cross(JE[0:3,twist_index_1],JE[0:3,twist_index_2])
    
    #print('v1_x_v2',v1_x_v2)
    v1_x_v2_norm = norm(v1_x_v2)
    
    d_v1_x_v2_dq = cross_product_gradient(twist_index_1, twist_index_2, JE,H,test_joint)
    
    d_v1_x_v2_norm_dq = cross_product_norm_gradient(twist_index_1,twist_index_2,JE,H,test_joint)
    
    
    numerator = d_v1_x_v2_dq*v1_x_v2_norm - (v1_x_v2*d_v1_x_v2_norm_dq)
    
    denom = v1_x_v2_norm**2
    #print('denom',denom)
    
    return numerator*(denom**(-1))
 
 
 ## Eq. 43. is here
 ## Function very specific to the paper and not generalized need to write a generalized sigmoid graident - refer to Philip's code-poly_gradient_fns for that
def sigmoid_gradient(twist_index_1,twist_index_2,twist_index_projected,JE,H,test_joint,sigmoid_slope):
    
    
    ### dsig(x)/dq = sig(x)(1-sig(x))dx/dq
    from numpy import cross,matmul,transpose
    from robot_functions import sigmoid
    
    v1 = JE[0:3,twist_index_1]
    v2 = JE[0:3,twist_index_2]
    
    ## Write exception case here to see if the twist index of normal and projected are not same
    

    
    vk = JE[0:3,twist_index_projected]
    
    n = cross(v1,v2)
   
    
    x = matmul(transpose(n),vk)
    
    dx_dq = normal_twist_projected_gradient(twist_index_1,twist_index_2,twist_index_projected,JE,H,test_joint)
    #dx_dq = matmul(d_nt_dq,vk) + matmul(transpose(n),d_vk_dq)
    
    #print('dx_dq',dx_dq)
    
    
        
    sigmoid_term = sigmoid(x,sigmoid_slope)*(1.0-sigmoid(x,sigmoid_slope))
    #print('sigmoid_term', sigmoid_term)
    
    return sigmoid_term*dx_dq