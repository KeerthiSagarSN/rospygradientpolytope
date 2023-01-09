# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 05:15:25 2022

@author: keerthi.sagar
"""

from numpy import array,sqrt,hstack,empty
import autograd.numpy as np_autograd
from autograd import hessian,grad,jacobian
from numpy.linalg import norm

## Only for 2-DOF as of now
def W_1(x,y):
    return np_autograd.array([[-x*((np_autograd.sqrt(x**2 + y**2)**(-1)))],\
                [-y*((np_autograd.sqrt(x**2 + y**2)**(-1)))]])


def W_2(x,y):
      
    return np_autograd.array([[-x*((np_autograd.sqrt(np_autograd.abs(H**2 - 2*H*y + x**2 + y**2))**(-1)))],\
                [(H-y)*((np_autograd.sqrt(np_autograd.abs(H**2 - 2*H*y + x**2 + y**2))**(-1)))]])



def W_3(x,y):   
         
    return np_autograd.array([[(L-x)*((np_autograd.sqrt(np_autograd.abs(H**2 - 2*H*y + L**2 - 2*L*x + x**2 + y**2))**(-1)))],\
                   [(H-y)*((np_autograd.sqrt(np_autograd.abs(H**2 - 2*H*y + L**2 - 2*L*x + x**2 + y**2))**(-1)))]])

def W_4(x,y):   
         
    return np_autograd.array([[(L-x)*((np_autograd.sqrt(np_autograd.abs(L**2 - 2*L*x + x**2 + y**2))**(-1)))],\
                   [(-y)*((np_autograd.sqrt(np_autograd.abs(L**2 - 2*L*x + x**2 + y**2))**(-1)))]])


def f_W_n(x,y):

    #L = 1
    #H = 1
    uv1 = np_autograd.array([[-x],[-y]])
    
    uv1_n = np_autograd.array([[-x*((np_autograd.sqrt(x**2 + y**2)**(-1)))],\
                   [-y*((np_autograd.sqrt(x**2 + y**2)**(-1)))]])

    
    
        
    uv2 = np_autograd.array([[-x],[H-y]])
    
    
    uv2_n = np_autograd.array([[-x*((np_autograd.sqrt(abs(H**2 - 2*H*y + x**2 + y**2))**(-1)))],\
                   [(H-y)*((np_autograd.sqrt(np_autograd.abs(H**2 - 2*H*y + x**2 + y**2))**(-1)))]])
    
    uv3 = np_autograd.array([[L-x],[H-y]])
    
    uv3_n = np_autograd.array([[(L-x)*((np_autograd.sqrt(np_autograd.abs(H**2 - 2*H*y + L**2 - 2*L*x + x**2 + y**2))**(-1)))],\
                   [(H-y)*((np_autograd.sqrt(np_autograd.abs(H**2 - 2*H*y + L**2 - 2*L*x + x**2 + y**2))**(-1)))]])
    
    uv4 = np_autograd.array([[L-x],[-y]])
    
    uv4_n = np_autograd.array([[(L-x)*((np_autograd.sqrt(np_autograd.abs(L**2 - 2*L*x + x**2 + y**2))**(-1)))],\
                   [(-y)*((np_autograd.sqrt(np_autograd.abs(L**2 - 2*L*x + x**2 + y**2))**(-1)))]])
    
    
    #W = hstack((uv1,uv2,uv3,uv4))
    
    return np_autograd.hstack((uv1_n,uv2_n,uv3_n,uv4_n))
def get_wrench_matrix(q,length,height):

    global L, H

    L = length
    H = height
    
    ##### Wrench matrix
    #x = 0.1
    #y = 0.15
    #L = 0.21
    #H = 0.61
    
    
    
    x = q[0]
    y = q[1]
    uv1 = array([[-x],[-y]])
    
    uv1_n = array([[-x*((sqrt(x**2 + y**2)**(-1)))],\
                   [-y*((sqrt(x**2 + y**2)**(-1)))]])

    
    
        
    uv2 = array([[-x],[H-y]])
    
    
    uv2_n = array([[-x*((sqrt(abs(H**2 - 2*H*y + x**2 + y**2))**(-1)))],\
                   [(H-y)*((sqrt(abs(H**2 - 2*H*y + x**2 + y**2))**(-1)))]])
    
    uv3 = array([[L-x],[H-y]])
    
    uv3_n = array([[(L-x)*((sqrt(abs(H**2 - 2*H*y + L**2 - 2*L*x + x**2 + y**2))**(-1)))],\
                   [(H-y)*((sqrt(abs(H**2 - 2*H*y + L**2 - 2*L*x + x**2 + y**2))**(-1)))]])
    
    uv4 = array([[L-x],[-y]])
    
    uv4_n = array([[(L-x)*((sqrt(abs(L**2 - 2*L*x + x**2 + y**2))**(-1)))],\
                   [(-y)*((sqrt(abs(L**2 - 2*L*x + x**2 + y**2))**(-1)))]])
    
    
    W = hstack((uv1,uv2,uv3,uv4))
    
    W_n = hstack((uv1_n,uv2_n,uv3_n,uv4_n))
    #############
    
    ## Autograd method

    H1 = jacobian(W_1)
    
    H2 = jacobian(W_2)
    H3 = jacobian(W_3)
    H4 = jacobian(W_4)

    
    Jx= jacobian(f_W_n,argnum=0)
    Jy= jacobian(f_W_n,argnum=1)
    
    
    
    



    #########
    '''
    ### Maple method
    d_W_dx = empty(shape = (2,4))
    
    d_W_dy = empty(shape = (2,4))
    
    
    
    d_W2_dx = empty(shape = (2,4))
    
    d_W2_dy = empty(shape = (2,4))
    
    ##############
    
    #d_W_dx[0,0] = (-y**2)*(((x**2 + y**2)**(3/2.0))**-1)

    d_W_dx[0,0] = -((x**2 + y**2)**-1) + 2*x**2 * ((x**2 + y**2)**-2)
    
    #d_W_dx[1,0] = (y*x)*(((x**2 + y**2)**(3/2.0))**-1)

    d_W_dx[1,0] = -y * (-2*x*(x**2 + y**2)**-2)
    
    

    
    
    
    #d_W_dy[0,0] = (y*x)*(((x**2 + y**2)**(3/2.0))**-1)
    ########### - First cable differentiation ###########
    d_W_dy[0,0] = -x * (-2*y*(x**2 + y**2)**-2)
    
    #d_W_dy[1,0] = -1*(x*x)*(((x**2 + y**2)**(3/2.0))**-1)
    
    d_W_dy[1,0] = -y * (-1*(x**2 + y**2)**-1) + (-1)*(x**2 + y**2)**-2
    


    ########### - Second cable differentiation ###########

    ### Differentiating with respect to x
    h1x = sqrt(abs(H**2 - 2*H*y + x**2 + y**2))
    
    

    d_W_dx[0,1] = -x * (-1.0*(h1x**-2))
    d_W_dx[1,1] = (H - y) * (-1.0*(h1x**-2))

    ### Differentiating with respect to y
    h1y = sqrt(abs(H**2 - 2*H*y + x**2 + y**2))

    d_W_dy[0,1] = -x * (-2.0*y*(h1y**-2))
    d_W_dy[1,1] = (H - y) * (-1.0*(h1y**-1)) + (-1.0)*(h1y**-2)
    ######## Third cable differentiation ###############

        ### Differentiating with respect to x
    h2x = sqrt(abs(H**2 - 2*H*y + L**2 - 2*L*x + x**2 + y**2))
    
    

    d_W_dx[0,2] = (L - x) * (-1.0*(h2x**-1)) + (-2*(L-x))*(h2x**-2)
    d_W_dx[1,2] = (H - y) * (-2.0*x*(h2x**-2))

    ### Differentiating with respect to y
    h2y = sqrt(abs(H**2 - 2*H*y + L**2 - 2*L*x + x**2 + y**2))

    d_W_dy[0,2] = (L - x) * (-2*y*(h2y**-2))
    d_W_dy[1,2] = (H - y) * (-1*(h2y**-1)) + (-1)*(h2y**-2)


    ######## Fourth cable differentiation ###############

        ### Differentiating with respect to x
    h3x = sqrt(abs(L**2 - 2*L*x + x**2 + y**2))
    
    

    d_W_dx[0,3] = (L - x) * (-1*(h3x**-1)) + (-2*(L-x))*(h3x**-2)
    d_W_dx[1,3] = (-y) * (-2*x*h3x**-2)

    ### Differentiating with respect to y
    h3y = sqrt(abs(L**2 - 2*L*x + x**2 + y**2))

    d_W_dy[0,3] = (L - x) * (-2*y*h3y**-2)
    d_W_dy[1,3] = (-y) * (-1*h3y**-1) + (-1)*(h3y**-2)

    '''
    ## Manual method
    '''


    
    ########## 
    
    denom_2 = H**2 - 2*H*y + x**2 + y**2
    signum_2 = (denom_2)*(abs(denom_2)**-1)
    
    d_W_dx[0,1] = (((x**2)*(signum_2)*denom_2) - abs(denom_2))*\
                        (((abs(denom_2))**(3/2.0))**-1)
    
    
    d_W_dx[1,1] = -1*(((H-y)*(x)*(signum_2)*denom_2))*\
                        (((abs(denom_2))**(3/2.0))**-1)
    
    
    
    d_W2_dx[0,1] = -1
    
    d_W2_dx[1,1] = 0
    
    
    
    d_W_dy[0,1] = -1*(((H-y)*(x)*(signum_2)*denom_2))*\
                        (((abs(denom_2))**(3/2.0))**-1)
    
    d_W_dy[1,1] = ((((H-y)**2)*(signum_2)*denom_2) - abs(denom_2))*\
                        (((abs(denom_2))**(3/2.0))**-1)
    
    
    d_W2_dy[0,1] = 0
    
    d_W2_dy[1,1] = -1
    
    ##########
    denom_3 = H**2 - 2*H*y + L**2 - 2*L*x + x**2 + y**2
    
    signum_3 = (denom_3)*(abs(denom_3)**-1)
    
    d_W_dx[0,2] = (((L-x)**2)*(signum_3)*(denom_3) - abs(denom_3))*\
                    (((abs(denom_3))**(3/2.0))**-1)
    
    
    
    d_W_dx[1,2] = (((H-y)*(L-x))*(signum_3)*(denom_3))*\
                    (((abs(denom_3))**(3/2.0))**-1)
    
    
    
    d_W2_dx[0,2] = -1
    
    d_W2_dx[1,2] = 0
    
    
    d_W_dy[0,2] = (((H-y)*(L-x))*(signum_3)*(denom_3))*\
                    (((abs(denom_3))**(3/2.0))**-1)
    
    
    d_W_dy[1,2] = (((H-y)**2)*(signum_3)*(denom_3) - abs(denom_3))*\
                    (((abs(denom_3))**(3/2.0))**-1)
    
    
    
    
    d_W2_dy[0,2] = 0
    
    d_W2_dy[1,2] = -1
    
    
    ###############
    denom_4 = L**2 - 2*L*x + x**2 + y**2
    
    signum_4 = (denom_4)*(abs(denom_4)**-1)
    
    d_W_dx[0,3] = (((L-x)**2)*(signum_4)*(denom_4) - abs(denom_4))*\
                    (((abs(denom_4))**(3/2.0))**-1)
                    
                    
    d_W_dx[1,3] = -1*(((L-x)*y)*(signum_4)*(denom_4))*\
                    (((abs(denom_4))**(3/2.0))**-1)
    
    
    
    
    d_W2_dx[0,3] = -1
    
    d_W2_dx[1,3] = 0
    
    
                    
    d_W_dy[0,3] = -1*(((L-x)*y)*(signum_4)*(denom_4))*\
                    (((abs(denom_4))**(3/2.0))**-1)
    
                    
    d_W_dy[1,3] = (((y)**2)*(signum_4)*(denom_4) - abs(denom_4))*\
                    (((abs(denom_4))**(3/2.0))**-1)
                    
    
    
    d_W2_dy[0,3] = 0
    
    d_W2_dy[1,3] = -1
    
    ### Return all wrench matrix and differentiation of wrench matrix with respect to x and y
    '''
    # NOt actually hessian - BUt jacobian for a 2D case
    Hessian_matrix = empty(shape = (2,4,2))
    #Hessian_matrix[:,:,0] = d_W_dx
    
    #Hessian_matrix[:,:,1] = d_W_dy
    
    Hessian_matrix[:,:,0] = Jx(x,y)
    Hessian_matrix[:,:,1] = Jy(x,y)
    return W,W_n,Hessian_matrix

'''
def main():

    q_inp = array([0.1,0.5])
    L = 1
    H = 1


    W,W_n,H = get_wrench_matrix(q_inp,L,H)
    print('W is',W)
    print('W_n is',W_n)
    
if __name__ == "__main__":
    main()
'''