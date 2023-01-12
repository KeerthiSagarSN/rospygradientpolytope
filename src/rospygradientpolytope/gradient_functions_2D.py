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
    
    
    Function for 2- DOF - Forces
    
    '''
    return H[0:2,twist_index,test_joint]
         
 
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
    
    return matmul(-skew(v2), dv1_dq) + matmul(skew(v1), dv2_dq)
    
    
    #return cross(dv1_dq,v2) + cross(v1,dv2_dq)
 
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
    
    v1_x_v2 = V_unit(cross(JE[0:3,twist_index_1],
                    JE[0:3,twist_index_2]))
    numerator = matmul(transpose(cross_product_gradient(twist_index_1, twist_index_2,JE,H, test_joint)),v1_x_v2)
    denom = sqrt(matmul(transpose(v1_x_v2),v1_x_v2))
    
    return numerator*(denom**(-1))
     
     
 
     


## Eq.44 full equation is here --> d_nT_vk_dq
## Function very specific to the paper - Not a generic function

def normal_twist_projected_gradient_2D(normal_index,twist_index,dn_dq,n,JE,H,test_joint):
    

    
    
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
    
        ### dsig(x)/dq = sig(x)(1-sig(x))dx/dq
    from numpy import cross,matmul,transpose,shape
    from robot_functions import sigmoid
    


    #for i in range(len(dn_dq)):


        

        
    vk = JE[0:2,twist_index]

    nT_vk = matmul(transpose(n[normal_index,:]),vk)
    
    x = nT_vk
    #n = V_unit(cross(v1,v2))
    #n = V_unit(cross(v1,v2))

    
    #x = matmul(transpose(n),vk)
    #x = nT_vk[normal_index]
    #print('shape of dn_dq',shape(dn_dq))
    #print('vk',shape(vk))

    #print('shape of n is',shape(n))

    #print('shape of H is',shape(H))

    dx_dq = matmul(dn_dq[:,normal_index,test_joint],vk) + matmul(transpose(n[normal_index,:]),H[:,twist_index,test_joint])
    #dx_dq = matmul(d_nt_dq,vk) + matmul(transpose(n),d_vk_dq)
    
    #print('dx_dq',dx_dq)
    
    
        

    #print('sigmoid_term', sigmoid_term)
    
    return dx_dq
    
    
 
 
 # Eq. 41 is here
 
def normal_gradient(H):
    
    from numpy import shape,zeros
    from copy import deepcopy
    
          
    '''
    
    Partial derivative of a normal vector - 2D 
   dn/dq = dW/dq


    
    Args


    JE - Jacobian
    H - Hessian
    
    Returns
    
    -------
    dn_dq
    
    -------
    
    '''
        
    #  (d(v1xv2)/dq)( ||v1 x v2||)  -  (v1 x v2)(d(|| v1 x v2 ||)/dq)/(|| v1 x v2 ||^2)
    dn_dq = deepcopy(H)
    

    #print('H is',H)
    #print('dn_dq',H[1,:,0])

    #input('test normal gradient')
    #print('H',H)
    #input('wait once')
    dn_dq[0,:,0] = -H[0,:,1]
    dn_dq[0,:,1] = H[0,:,0]

    dn_dq[1,:,0] = -H[1,:,1]
    dn_dq[1,:,1] = H[1,:,0]
    
    #print('denom',denom)
    
    return dn_dq
 
 
 ## Eq. 43. is here
 ## Function very specific to the paper and not generalized need to write a generalized sigmoid graident - refer to Philip's code-poly_gradient_fns for that
 # 2D sigmoid gradient
def sigmoid_gradient(normal_index,twist_index,dn_dq,n,JE,H,test_joint,sigmoid_slope):
    
    
    ### dsig(x)/dq = sig(x)(1-sig(x))dx/dq
    from numpy import cross,matmul,transpose
    from robot_functions import sigmoid
    


    #for i in range(len(dn_dq)):


        

        
    vk = JE[0:2,twist_index]

    nT_vk = matmul(transpose(n[normal_index,:]),vk)
    
    x = nT_vk
    #n = V_unit(cross(v1,v2))
    #n = V_unit(cross(v1,v2))

    
    #x = matmul(transpose(n),vk)
    #x = nT_vk[normal_index]
    
    dx_dq = matmul(dn_dq[:,normal_index,test_joint],vk) + matmul(transpose(n[normal_index,:]),H[:,twist_index,test_joint])
    
    #dx_dq = matmul(d_nt_dq,vk) + matmul(transpose(n),d_vk_dq)
    
    #print('dx_dq',dx_dq)
    
    
        
    sigmoid_term = sigmoid(x,sigmoid_slope)*(1.0-sigmoid(x,sigmoid_slope))
    #print('sigmoid_term', sigmoid_term)
    
    return sigmoid_term*dx_dq

'''
def normal_qr_gradient(Wm):
    # Compute the gradient of the QR decomposition with respect to W
    # Input the wrench matrix and compute the gradient here
    from numpy import shape,zeros,array
    from scipy.linalg import qr
    
    dn_dq = zeros(shape = (shape(Wm)[1],2))
    
    

    

    for ij in range(shape(Wm)[1]):

        #W = array([Wm[:,j]])
        

        Wmk = transpose(array([Wm[:,ij]]))

        Q, R = qr(Wmk)
        print('n_k inside the loop',Q)
        m, n = Wmk.shape
        #n = 1
        dQ = zeros((m, n, m, n))
        dR = zeros((m, n, m, n))
        for i in range(m):
            for j in range(n):
                for k in range(m):
                    for l in range(n):
                        if k == i and l == j:
                            dR[i, j, k, l] = 1
                        elif k == i:
                            dR[i, j, k, l] = -Q[i, l] / R[j, j]
                        elif l == j:
                            dR[i, j, k, l] = Q[k, j] / R[j, j]
                        else:
                            dR[i, j, k, l] = 0
        for i in range(m):
            for j in range(n):
                for k in range(m):
                    for l in range(n):
                        dQ[i, j, k, l] = sum(dR[i, j, :, l] * Q[k, :])

        print('n_k in QR decomp is',Q[:,-1])
        print('dQ',dQ)
        dQ_n = dQ[:,-1,-1,-1]
        print('dQ_n',dQ_n)
        dn_dq[ij,:] = dQ_n
        print('dn_dq[ij,:]',dn_dq[ij,:])
        
        print('dn_dq',dn_dq)
        input('wait here')
        
    return dn_dq
'''

def normal_qr_gradient(Wm,H):
    # Compute the gradient of the QR decomposition with respect to W
    # Input the wrench matrix and compute the gradient here
    from numpy import shape,zeros,array,triu,outer,eye
    from numpy.linalg import pinv
    from scipy.linalg import qr
    
    dn_dq = zeros(shape = (shape(Wm)[0],shape(Wm)[1],2))
    
    
    for dq in range(shape(Wm)[0]):
    

        for ij in range(shape(Wm)[1]):

            #W = array([Wm[:,j]])
            
            '''
            Wmk = transpose(array([Wm[:,ij]]))
            print('Wmk',Wmk)
            Q, R = qr(Wmk)
            print('Q is',Q)
            Q = Q[:,-1]
            print('n_k inside the loop',Q)
            R_inv = pinv(R)
            
            # Compute gradient of R with respect to A
            R_grad = triu(outer(R, R_inv) - eye(Q.shape[0]))
            
            # Compute gradient of Q with respect to A
            Q_grad = Q @ (eye(Q.shape[0]) - R_inv @ R_grad)

            print('Q_grad',Q_grad)
            input('wait here')
            '''
            ## Skew2D format - here too - Similar to scipy.qr decompsition method
            sign_x = Wm[0,ij]
            sign_y = Wm[1,ij]

            #print('testing ij',ij)
            if ((sign_x > 1) and (sign_y > 1)) or ((sign_x < 1) and (sign_y < 1)):

                dn_dq[0,ij,dq] = -H[1,ij,dq]
                dn_dq[1,ij,dq] = H[0,ij,dq]

            
            if ((sign_x > 1) and (sign_y <1)) or ((sign_x < 1) and (sign_y > 1)):
                dn_dq[0,ij,dq] = -H[1,ij,dq]
                dn_dq[1,ij,dq] = H[0,ij,dq]






    #print('dn_dq',dn_dq)
    
        
    return dn_dq