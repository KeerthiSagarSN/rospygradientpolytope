# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:46:19 2022

@author: keerthi.sagar
"""

import polytope as pc

from numpy import shape,arange,array,zeros,cross,count_nonzero,transpose,matmul,dot,zeros
import polytope
import rospygradientpolytope.robot_functions as robot_functions
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import itertools
from rospygradientpolytope.linearalgebra import V_unit,check_ndarray
from numpy import unravel_index,argmax,min,hstack,vstack,argwhere
import rospy




def get_Cartesian_polytope(jacobian, joint_space_vrep):
    Pv = zeros([shape(joint_space_vrep)[0], shape(jacobian)[0]])

    for row, i in zip(joint_space_vrep, range(shape(joint_space_vrep)[0])):
        Pv[i, :] = matmul(jacobian, row)
    return Pv


def plot_polytope_3d(poly):
    V = polytope.extreme(poly)
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.gca()
    hull = ConvexHull(V, qhull_options='Qs QJ')
    ax.plot(hull.points[hull.vertices, 0],
            hull.points[hull.vertices, 1],
            hull.points[hull.vertices, 2], 'ko', markersize=4)

    s = ax.plot_trisurf(hull.points[:, 0], hull.points[:, 1], hull.points[:, 2], triangles=hull.simplices,
                        color='red', alpha=0.2, edgecolor='k')

    plt.show()
    return ax


def get_polytope_hyperplane(JE,active_joints,cartesian_dof_input,qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope):

    from numpy import unravel_index,argmax,min,zeros,count_nonzero
    ### Declarations here 

    ## Import robot
    

    deltaqq = qdot_max - qdot_min

    

    ### Cartesian degrees of freedom - Mostly 3 - Velocity
    ## Input a 6x1^T vector - [vx=True,vy=True,vz=True,wx=False,wy=False,wz = False]
    ## This is amasked array
    cartesian_dof_mask = cartesian_dof_input
    cartesian_dof = count_nonzero(cartesian_dof_mask)

    ## Listing out joints like Philip's joint
    #active_joints = arange(cartesian_dof)
    #active_joints = shape(JE)[1]


    ### Import from Philip s function- robot_functions

    Nmatrix, Nnot = robot_functions.getDofCombinations(arange(active_joints), cartesian_dof)




    #print('Nmatrix is',Nmatrix)

    #print('self.Nnot is',Nnot)
    number_of_combinations = shape(Nmatrix)[0]

    # Generalized cross-product need to be implemented in the future
    # Here for Vector only of size - 3 is implemented in this code
    #print('self.cartesian_dof',self.cartesian_dof)
    ### Screw is represented as : $ = [v w] from the jacobian


    v_k = zeros(shape = (cartesian_dof,active_joints))


    v_k = JE[0:3,:]
    #print('v_k',self.v_k)

    # Eq,14
    ## Initialize all normals here - of size:
    ## Generalized normal should be impleemented in the future
    ## n_k = number_of_rows(Nmatrix)x3

    n_k = zeros(shape = (shape(Nmatrix)[0],3))

    
    # n- (m-1) twists ; n- no.of dof, m = 2 (1 - (2-1)) - 1
    # Compute all the normals n_k - Eq 15
    ## Compute all normals here
    for i in range(len(n_k)):


        n_k[i] = (V_unit(cross(check_ndarray(v_k[:,Nmatrix[i,0]]), check_ndarray(v_k[:,Nmatrix[i,1]]))))
        #print('n_k is',self.n_k)
        ## Compute all projection vertices here
        ## Equation 16 in paper is here
        ## Instantiate all projection vertices here
        ## l_k = n_k x  v_k x 3


    l_k = zeros(shape = (len(n_k),shape(Nnot)[1]))



    for i in range(len(n_k)):
        for j in range(shape(Nnot)[1]):


         ### Should i be taking the unit vector here ??????

         #self.l_k[i,j] = transpose(dot(self.n_k[i,:],V_unit(self.v_k[:,self.Nnot[i,j]])))
         l_k[i,j] = transpose(dot(n_k[i,:],v_k[:,Nnot[i,j]]))

     #print('l_k is',self.l_k)

     ### Instantiate h+ parameter

    h_plus = zeros(shape=(len(n_k)))
    h_minus = zeros(shape=(len(n_k)))

    h_plus_hat = zeros(shape=(len(n_k)))
    h_minus_hat = zeros(shape=(len(n_k)))


     ## Eq.19 and Eq.20 from paper here
     ## Sigmoid function from Philip's robot_functions.py
     ########### Actual hyperplane parameters

    for i in range(len(n_k)):
        for j in range(shape(Nnot)[1]):

            h_plus[i] = h_plus[i] + max(0,(deltaqq[Nnot[i,j]]*(l_k[i,j])))


            h_minus[i] = h_minus[i] + min(array([0,(deltaqq[Nnot[i,j]]*(l_k[i,j]))]))

     #print(self.deltaqq[self.Nnot[i,j]]*(self.l_k[i,j]))





    ########## Estimated hyperplane parameters

    for i in range(len(n_k)):
        for j in range(shape(Nnot)[1]):

         #print('self.l_k dot product is',self.l_k[i,j])

         h_plus_hat[i] = h_plus_hat[i] + robot_functions.sigmoid(l_k[i,j],sigmoid_slope)*deltaqq[Nnot[i,j]]*(l_k[i,j])

         h_minus_hat[i] = h_minus_hat[i] + robot_functions.sigmoid(l_k[i,j],-sigmoid_slope)*deltaqq[Nnot[i,j]]*(l_k[i,j])

         '''
         print('robot_functions.sigmoid(self.l_k[i,j],sigmoid_slope)',robot_functions.sigmoid(self.l_k[i,j],sigmoid_slope))
         print('robot_functions.sigmoid(self.l_k[i,j],-sigmoid_slope)',robot_functions.sigmoid(self.l_k[i,j],-sigmoid_slope))
         print(' self.h_plus_hat[i] is', self.h_plus_hat[i])
         print(' self.h_minus_hat[i] is', self.h_minus_hat[i])
         input('wait here')
         '''

    #print('h_plus_i is',self.h_plus)
    #print('h_minus_i is',self.h_minus)
    #print('h_plus_hat',self.h_plus_hat)
    #print('h_minus_hat',self.h_minus_hat)

    #input('wait here')



    # Eq . 23 here
    ## ALl vertices on the hyper-plane are defined below - p_plus points on the

    #### ACtual parameters here
    p_plus = zeros(shape = (len(n_k),3))

    p_minus = zeros(shape = (len(n_k),3))


    ########## Estimated parameters are here

    p_plus_hat = zeros(shape = (len(n_k),3))

    p_minus_hat = zeros(shape = (len(n_k),3))


    ##### Actual parameters are here

    for i in range(len(l_k)):

        ## I have transposed here for qdot_min which s only becos of the input format- need to generalize it
        p_plus[i,:] = h_plus[i]*n_k[i,:]  + transpose(matmul(JE[0:3,:],transpose(array([qdot_min]))))

        p_minus[i,:] = h_minus[i]*n_k[i,:]  + transpose(matmul(JE[0:3,:],transpose(array([qdot_min]))))


    ##### Estimated parameters are here
        
    for i in range(len(l_k)):
        #p_plus = h_plus[i]*n[i,:] + matmul(JE,qmin)
        #p_plus = np.vstack((p_plus,h_plus[i]*n[i,:]))

        ## I have transposed here for qdot_min which s only becos of the input format- need to generalize it
        p_plus_hat[i,:] = h_plus_hat[i]*n_k[i,:]  + transpose(matmul(JE[0:3,:],transpose(array([qdot_min]))))
        #p_minus = h_minus[i]*n[i,:] + matmul(JE,qmin)
        p_minus_hat[i,:] = h_minus_hat[i]*n_k[i,:]  + transpose(matmul(JE[0:3,:],transpose(array([qdot_min]))))


    '''
    print('p_plus',p_plus)
    print('p_minus is',p_minus)
    print('p_plus_hat is',p_plus_hat)
    print('p_minus_hat',p_minus_hat)

    print('p_plus',h_plus)
    print('p_minus is',h_minus)
    print('p_plus_hat is',h_plus_hat)
    print('p_minus_hat',h_minus_hat)
    '''
    #input('wait here')

    return h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot


## Take into consideration obstacle constraints
## return AX <= B - H-rep polytope for joint polytope with environmental constraints
def get_constraint_obstacle_joint_polytope(JE,obstacle_link_vector,danger_parameter):

    from numpy import matmul,size,ndarray
    from numpy.linalg import norm

    # n- active links on the Control points
    # Obstacle_link_matrix - n x 3 
    # JE - Jacobian matrix - 3 x n - Only considering translational velocity 
    
    # Active qdot - is n x 1
    # AX <= B -> Jok*qdot <= b0
    # v_obs*JE*qdot <= danger_distance*norm(v_obs) -> (12 x 3 ) * (3x12) * (12 x 1) <= (12 x 1)

    active_joints = shape(JE)[1]

    #active_joints = 6
    v_obs = obstacle_link_vector
    print('shape(v_obs)',shape(v_obs))
    #A_obs = matmul(v_obs,JE[0:3,:])
    A_obs = zeros(shape=(1,active_joints))


    
    B_obs = zeros(active_joints)

    for i in range(active_joints):
        # Meaning KUKA joints are 
        if i < 6:
            norm_dist = norm(v_obs[i,:])
            JE_temp = zeros(shape=(3,active_joints))
            #A_obs[i,i] = matmul(v_obs[i,:]*(norm_dist**(-1)),JE[0:3,i])
            #A_obs[i,i] = matmul(v_obs[i,:]*(norm_dist**(-1)),JE[0:3,i])
            print('JE is',JE)
            if i == 0:
                JE_temp[:,0] = JE[0:3,0] 
            else:
                JE_temp[:,0:i+1] = JE[0:3,0:i+1] 

            A_obs = vstack((A_obs,matmul(array([V_unit(v_obs[i,:])]),JE_temp)))
            norm_dist = norm(v_obs[i,:])
            print('norm_dist',norm_dist)
            B_obs[i] = danger_parameter*(norm_dist**(2)) - norm_dist
        else:
            JE_temp = zeros(shape=(3,active_joints))
            JE_temp[:,6:i+1] = JE[0:3,6:i+1] 
            norm_dist = norm(v_obs[i,:])
            A_obs = vstack((A_obs,matmul(array([V_unit(v_obs[i,:])]),JE_temp)))
            norm_dist = norm(v_obs[i,:])
            print('norm_dist',norm_dist)
            B_obs[i] = danger_parameter*(norm_dist**(2)) - norm_dist
        print('joint i',i)
        print('JE_temp',JE_temp)  
        #input('stop now')
    A_obs = A_obs[1:,:]
    # print('A_obs', A_obs)
    # print('B_obs', B_obs)
    B_obs = ndarray.flatten(B_obs)

    # print('shape(A_obs)',shape(A_obs))
    # print('shape(B_obs)',transpose([B_obs]))
    return -A_obs, B_obs


def get_constraint_obstacle_jacobian(JE,obstacle_link_vector,danger_parameter):

    from numpy import matmul,size,ndarray
    from numpy.linalg import norm

    # n- active links on the Control points
    # Obstacle_link_matrix - n x 3 
    # JE - Jacobian matrix - 3 x n - Only considering translational velocity 
    
    # Active qdot - is n x 1
    # AX <= B -> Jok*qdot <= b0
    # v_obs*JE*qdot <= danger_distance*norm(v_obs) -> (12 x 3 ) * (3x12) * (12 x 1) <= (12 x 1)

    active_joints = shape(JE)[1]

    #active_joints = 6
    v_obs = obstacle_link_vector
    print('shape(v_obs)',shape(v_obs))

    # Collision jacobian is
    #J_coll = zeros(shape=(1,active_joints))
    #A_obs = matmul(v_obs,JE[0:3,:])
    A_obs = zeros(shape=(1,active_joints))
    
    B_obs = zeros(active_joints)

    for i in range(active_joints):
        # Meaning KUKA joints are 
        if i < 6:
            # norm_dist = norm(v_obs[i,:])
            JE_temp = zeros(shape=(3,active_joints))
            #A_obs[i,i] = matmul(v_obs[i,:]*(norm_dist**(-1)),JE[0:3,i])
            #A_obs[i,i] = matmul(v_obs[i,:]*(norm_dist**(-1)),JE[0:3,i])
            print('JE is',JE)
            if i == 0:
                JE_temp[:,0] = JE[0:3,0] 
            else:
                JE_temp[:,0:i+1] = JE[0:3,0:i+1] 

            A_obs = vstack((A_obs,matmul(array([V_unit(v_obs[i,:])]),JE_temp)))
            # norm_dist = norm(v_obs[i,:])
            # print('norm_dist',norm_dist)
            # B_obs[i] = danger_parameter*(norm_dist**(2)) - norm_dist
        else:
            JE_temp = zeros(shape=(3,active_joints))
            JE_temp[:,6:i+1] = JE[0:3,6:i+1] 
            # norm_dist = norm(v_obs[i,:])
            A_obs = vstack((A_obs,matmul(array([V_unit(v_obs[i,:])]),JE_temp)))
            # norm_dist = norm(v_obs[i,:])
            # print('norm_dist',norm_dist)
            # B_obs[i] = danger_parameter*(norm_dist**(2)) - norm_dist
        print('joint i',i)
        print('JE_temp',JE_temp)  
        #input('stop now')
    A_obs = A_obs[1:,:]
    # print('A_obs', A_obs)
    # print('B_obs', B_obs)
    # B_obs = ndarray.flatten(B_obs)
    J_coll = A_obs
    # print('shape(A_obs)',shape(A_obs))
    # print('shape(B_obs)',transpose([B_obs]))
    return J_coll 



# Include joint limits constraints
def get_constraint_joint_limit_polytope(JE,q,active_joints,qdot_min,qdot_max,q_min,q_max,q_mean,psi_max, psi_min):
    from numpy import unravel_index,argmax,min,zeros,count_nonzero,max,matmul,identity,size, ndarray
    from numpy.linalg import norm
    from pypoman import compute_polytope_vertices

    
    psi_max_qdot_max = psi_max
    psi_min_qdot_min = psi_min
    for i in range(active_joints):

       
        #print('q_mean[i]',q_mean[i])
        #print('q[i]',q[i])
        if q_mean[i] > q[i]:
            d_max = q_mean[i]
        else:
            d_max = q[i]
        #print('d_max',d_max)
        if q_mean[i] < q[i]:
            d_min = q_mean[i]
        else:
            d_min = q[i]

        
        #print('d_min',d_min)

        psi_max_qdot_max[i] = (1.0 - ((d_max - q_mean[i])/(q_max[i] - q_mean[i]))**2)*qdot_max[i]
        
        psi_min_qdot_min[i] = (1.0 - ((d_min - q_mean[i])/(q_min[i] - q_mean[i]))**2)*qdot_min[i]
    #print('psi_max', psi_max)
    #print('qdot_max',qdot_max)
    #print('qdot_min',qdot_min)
    B_jpl = hstack((psi_max_qdot_max,-1.0*psi_min_qdot_min))
    A_jpl = hstack((identity(active_joints),-1.0*identity(active_joints)))
    
    
    return transpose(A_jpl), transpose(B_jpl)


# Include joint limits and collision avoidance constraints
def get_constraint_polytope(JE,q,active_joints,qdot_min,qdot_max,q_min,q_max,q_mean,psi_max, psi_min,\
                            obstacle_link_vector,obstacle_dist_vector,danger_parameter):
    from numpy import unravel_index,argmax,min,zeros,count_nonzero,max,matmul,identity,size, ndarray, around
    from numpy.linalg import norm
    from pypoman import compute_polytope_vertices

    #active_joints = 6
    # qdot_min = qdot_min[0:active_joints]
    # qdot_max = qdot_max[0:active_joints]
    # q_min = q_min[0:active_joints]
    # q_max = q_max[0:active_joints]
    # psi_min = psi_min[0:active_joints]
    # psi_max = psi_max[0:active_joints]
    v_obs = obstacle_link_vector
    v_obs.dtype = float

    

    #print('shape(v_obs)',shape(v_obs))
    #A_obs = matmul(v_obs,JE[0:3,:])

    #time_begin = rospy.Time.now()
    A_obs = zeros(shape=(active_joints,active_joints),dtype="float64")
    

    
    B_obs = zeros(active_joints,dtype="float64")
    #danger_parameter = 1.0

    for i in range(active_joints):
        # Meaning KUKA joints are 
        if i < 6:
            #norm_dist = norm(v_obs[i,:])
            JE_temp = zeros(shape=(3,active_joints))
            #A_obs[i,i] = matmul(v_obs[i,:]*(norm_dist**(-1)),JE[0:3,i])
            #A_obs[i,i] = matmul(v_obs[i,:]*(norm_dist**(-1)),JE[0:3,i])
            #print('JE is',JE)
            if i == 0:
                JE_temp[:,0] = JE[0:3,0] 
            else:
                JE_temp[:,0:i+1] = JE[0:3,0:i+1] 

            #A_obs = vstack((A_obs,matmul(array([V_unit(v_obs[i,:])]),JE_temp)))
            A_obs[i,:] = matmul(v_obs[i,:]/obstacle_dist_vector[i],JE_temp)

            #norm_dist = norm(v_obs[i,:])
            #print('norm_dist',norm_dist)
            
            B_obs[i] = danger_parameter*(obstacle_dist_vector[i]**(2)) - obstacle_dist_vector[i]
        else:
            JE_temp = zeros(shape=(3,active_joints))
            JE_temp[:,6:i+1] = JE[0:3,6:i+1] 
            #norm_dist = norm(v_obs[i,:])
            # A_obs = vstack((A_obs,matmul(array([V_unit(v_obs[i,:])]),JE_temp)))
            A_obs[i,:] = matmul(v_obs[i,:]/obstacle_dist_vector[i],JE_temp)

            #norm_dist = norm(v_obs[i,:])
            #print('norm_dist',norm_dist)
            B_obs[i] = danger_parameter*(obstacle_dist_vector[i]**(2)) - obstacle_dist_vector[i]
        #print('joint i',i)
        #print('JE_temp',JE_temp)  
        #input('stop now')
    #A_obs = A_obs[1:,:]
    # print('A_obs', A_obs)
    #print('B_obs', B_obs)
    B_obs = ndarray.flatten(B_obs)

    B_obs = transpose(array([B_obs]))
    psi_max_qdot_max = psi_max
    psi_min_qdot_min = psi_min
    for i in range(active_joints):

       
        #print('q_mean[i]',q_mean[i])
        #print('q[i]',q[i])
        if q_mean[i] > q[i]:
            d_max = q_mean[i]
        else:
            d_max = q[i]
        #print('d_max',d_max)
        if q_mean[i] < q[i]:
            d_min = q_mean[i]
        else:
            d_min = q[i]

        
        #print('d_min',d_min)

        psi_max_qdot_max[i] = (1.0 - ((d_max - q_mean[i])/(q_max[i] - q_mean[i]))**2)*qdot_max[i]
        
        psi_min_qdot_min[i] = (1.0 - ((d_min - q_mean[i])/(q_min[i] - q_mean[i]))**2)*qdot_min[i]
    #print('psi_max', psi_max)
    #print('qdot_max',qdot_max)
    #print('qdot_min',qdot_min)
    B_jpl = hstack((psi_max_qdot_max,-psi_min_qdot_min))
    A_jpl = hstack((identity(active_joints),-identity(active_joints)))
    #print('A_jpl',A_jpl)
    #print('B_jpl',B_jpl)

    #print('A_obs',A_obs)
    #print('B_obs',B_obs)


   
    A_cmp = vstack((transpose(A_jpl),-A_obs))
    B_cmp = vstack((transpose(array([B_jpl])),B_obs))
    #print('B_jpl',size(B_jpl))

    A_cmp.dtype = 'float64'
    B_cmp.dtype = 'float64'
    #print('shape(A_jpl)',shape(A_jpl))
    #print('shape(B_jpl)',shape(B_jpl))
    #print('duration is', rospy.Time.now()- time_begin)
    
    return A_cmp, B_cmp





def get_cartesian_polytope_hyperplane(JE,active_joints,qdot_min,qdot_max):

    from numpy import unravel_index,argmax,min,zeros,count_nonzero
    ### Declarations here 

    ## Import robot
    qdot_max = qdot_max[0:active_joints]
    qdot_min = qdot_min[0:active_joints]

    deltaqq = qdot_max - qdot_min

    

    ### Cartesian degrees of freedom - Mostly 3 - Velocity
    ## Input a 6x1^T vector - [vx=True,vy=True,vz=True,wx=False,wy=False,wz = False]
    ## This is amasked array
    # Saving some computaiton time
    #cartesian_dof_mask = cartesian_dof_input
    #cartesian_dof = count_nonzero(cartesian_dof_mask)
    cartesian_dof = 3
    ## Listing out joints like Philip's joint
    #active_joints = arange(cartesian_dof)
    #active_joints = shape(JE)[1]


    ### Import from Philip s function- robot_functions

    Nmatrix, Nnot = robot_functions.getDofCombinations(arange(active_joints), cartesian_dof)




    #print('Nmatrix is',Nmatrix)

    #print('self.Nnot is',Nnot)
    number_of_combinations = shape(Nmatrix)[0]
    print('number of combination',number_of_combinations)

    # Generalized cross-product need to be implemented in the future
    # Here for Vector only of size - 3 is implemented in this code
    #print('self.cartesian_dof',self.cartesian_dof)
    ### Screw is represented as : $ = [v w] from the jacobian


    #v_k = zeros(shape = (cartesian_dof,active_joints))


    v_k = JE[0:3,:]
    #print('v_k',self.v_k)

    # Eq,14
    ## Initialize all normals here - of size:
    ## Generalized normal should be impleemented in the future
    ## n_k = number_of_rows(Nmatrix)x3

    n_k = zeros(shape = (shape(Nmatrix)[0],3))

    
    # n- (m-1) twists ; n- no.of dof, m = 2 (1 - (2-1)) - 1
    # Compute all the normals n_k - Eq 15
    ## Compute all normals here
    for i in range(len(n_k)):


        n_k[i] = (V_unit(cross(check_ndarray(v_k[:,Nmatrix[i,0]]), check_ndarray(v_k[:,Nmatrix[i,1]]))))
        


    l_k = zeros(shape = (len(n_k),shape(Nnot)[1]))



    for i in range(len(n_k)):
        for j in range(shape(Nnot)[1]):


         ### Should i be taking the unit vector here ??????

         #self.l_k[i,j] = transpose(dot(self.n_k[i,:],V_unit(self.v_k[:,self.Nnot[i,j]])))
         l_k[i,j] = transpose(dot(n_k[i,:],v_k[:,Nnot[i,j]]))

     #print('l_k is',self.l_k)

     ### Instantiate h+ parameter

    h_plus = zeros(shape=(len(n_k)))
    h_minus = zeros(shape=(len(n_k)))

    ########### Actual hyperplane parameters

    for i in range(len(n_k)):
        for j in range(shape(Nnot)[1]):

            h_plus[i] = h_plus[i] + max(0,(deltaqq[Nnot[i,j]]*(l_k[i,j])))


            h_minus[i] = h_minus[i] + min(array([0,(deltaqq[Nnot[i,j]]*(l_k[i,j]))]))

   
    ## ALl vertices on the hyper-plane are defined below - p_plus points on the

    #### ACtual parameters here
    p_plus = zeros(shape = (len(n_k),3))

    p_minus = zeros(shape = (len(n_k),3))



    ##### Actual parameters are here

    for i in range(len(l_k)):

        ## I have transposed here for qdot_min which s only becos of the input format- need to generalize it
        p_plus[i,:] = h_plus[i]*n_k[i,:]  + transpose(matmul(JE[0:3,:],transpose(array([qdot_min]))))

        p_minus[i,:] = h_minus[i]*n_k[i,:]  + transpose(matmul(JE[0:3,:],transpose(array([qdot_min]))))





    return p_plus,p_minus,n_k




def get_capacity_margin(JE,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints,cartesian_dof_input,qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope):
    #### Capacity Margin

    ### Vertices of desired polytope: Input from user
    v_k_d = cartesian_desired_vertices

    ### Eq 26, 27, 28 - Calculating capacity margin from the hyperplane parameters
    ## Distance between each desired vertex and the base  polytope is recorded as an array called capacity margin
    ## Gamma_plus is of size fxl - where f is the facets of the velocity polytope and and l is the number of vertices
    ## of the desired twist/wrench polytope

    ##### Actual parameters are here
    Gamma_plus = zeros([len(n_k),len(v_k_d)])

    Gamma_minus = zeros([len(n_k),len(v_k_d)])

    ##### Estimatedparameters are here
    Gamma_plus_hat = zeros([len(n_k),len(v_k_d)])

    Gamma_minus_hat = zeros([len(n_k),len(v_k_d)])

    Gamma_array_counter = 0
    ### Calculating Gamma_plus here
    #JE = self.jacobian_hessian
    # Eq. 26 calculated here - Gamma_plus and Gamma_minus
    ##### Actual parameters are here
    #for facets in range(len(self.n_k)):
    ### Screw is represented as : $ = [v w] from the jacobian
    cartesian_dof_mask = cartesian_dof_input
    cartesian_dof = count_nonzero(cartesian_dof_mask)
    v_k = zeros(shape = (cartesian_dof,active_joints))

    #print('cartesian_dof_mask',cartesian_dof_mask)
    #print('JE',JE)
    v_k = JE[cartesian_dof_mask,:]
    JE = v_k

    Gamma_plus_LHS = transpose(array([h_plus])) + matmul(matmul(n_k,JE),transpose(array([qdot_min])))


    #input('wait here')
    Gamma_plus_hat_LHS = transpose(array([h_plus_hat])) + matmul(matmul(n_k,JE),transpose(array([qdot_min])))

    Gamma_minus_LHS = transpose(array([h_minus])) + matmul(matmul(n_k,JE),transpose(array([qdot_min])))

    Gamma_minus_hat_LHS = transpose(array([h_minus_hat])) + matmul(matmul(n_k,JE),transpose(array([qdot_min])))

    ## Only for testing finding the minimum in the loop

    for vertex in range(len(v_k_d)):
        #print 'vertex'
        #print vertex
        #self.Gamma_plus[facets,vertex] = abs(matmul(transpose(self.n_k[facets,:]),self.p_plus[facets,:]) - matmul(transpose(self.n_k[facets,:]),self.v_k_d[vertex,:]))

        '''
        print('self.n_k',self.n_k)
        print('transpose((self.v_k_d[vertex,:]))', transpose((array([self.v_k_d[vertex,:]]))))
        print('Gamma_plus_LHS',Gamma_plus_LHS)

        print('matmul(self.n_k, transpose((array([self.v_k_d[vertex,:]]))))',matmul(self.n_k, transpose((array([self.v_k_d[vertex,:]])))))
        '''
        Gamma_plus[:,vertex] =  transpose(Gamma_plus_LHS  - matmul(n_k, transpose((array([v_k_d[vertex,:]])))))

        
        ## Estimated parameters are here
        #self.Gamma_plus_hat[facets,vertex] = abs(matmul(transpose(self.n_k[facets,:]),self.p_plus_hat[facets,:]) - matmul(transpose(self.n_k[facets,:]),self.v_k_d[vertex,:]))

        Gamma_minus[:,vertex] = transpose(Gamma_minus_LHS  - matmul(n_k, transpose((array([v_k_d[vertex,:]])))))
        #self.Gamma_plus[facets,vertex]= hstack((self.Gamma_plus,transpose(array([h_plus]))+ matmul(matmul(n, transpose(JE)), qmin) - matmul(n, transpose(desired_vertices[vertex, :]))))
        #self.Gamma_minus[facets,vertex]= hstack((Gamma_minus,transpose(array([h_minus]))+ matmul(matmul(-n, transpose(JE)), qmin) - matmul(-n, transpose(desired_vertices[vertex, :]))))
        #self.Gamma_minus[facets,vertex] = abs(matmul(transpose(self.n_k[facets,:]),self.p_minus[facets,:]) - matmul(-1*transpose(self.n_k[facets,:]),self.v_k_d[vertex,:]))

        Gamma_plus_hat[:,vertex] =  transpose(Gamma_plus_hat_LHS  - matmul(n_k, transpose((array([v_k_d[vertex,:]])))))

        ### estimated parameteres are here
        Gamma_minus_hat[:,vertex] = transpose(Gamma_minus_hat_LHS  - matmul(n_k, transpose((array([v_k_d[vertex,:]])))))



    #print('Gamma_plus is',self.Gamma_plus)
    #input('stop here')
    shape_gamma = shape(Gamma_plus)
    Gamma_sum = zeros(shape=(shape_gamma[0],shape_gamma[1],2))
    Gamma_sum[:,:,0] = Gamma_plus_hat
    Gamma_sum[:,:,1] = -Gamma_minus_hat

    ## Eq.30 is here

    Gamma_plus_flat = check_ndarray(Gamma_plus)
    Gamma_minus_flat = check_ndarray(Gamma_minus)


    ### Estimated parameters are here
    Gamma_plus_flat_hat = check_ndarray(Gamma_plus_hat)
    Gamma_minus_flat_hat = check_ndarray(Gamma_minus_hat)

    ## Exact representation as in Eq. 31
    #print('len(self.Gamma_plus_flat)',len(self.Gamma_plus_flat))
    #input('wait here')
    Gamma_total = hstack((Gamma_plus_flat,-1*Gamma_minus_flat))


    ### Estimated parameters are here

    Gamma_total_hat = hstack((Gamma_plus_flat_hat,-1*Gamma_minus_flat_hat))


    #print('self.Gamma_total',self.Gamma_total)
    #print('self.Gamma_total_hat',self.Gamma_total_hat)


    ## eq. 31 is here
    # Using the minimum function which is not continous here
    Gamma_min = min(Gamma_total)

    #print('self.Gamma_min',self.Gamma_min)

    ### Very bad implementation - Need to optimize - TODO
    ## Row of facet_pair_idx refers to Normal of the capacity margin plane
    ## Coluimn of facet_pair_idx refers to Point on the desired polytope closest to the capacity margin plane
    if min(Gamma_plus_hat) < -1*min(Gamma_minus_hat):
        facet_pair_idx = argwhere(1.0*Gamma_plus_hat == min(1.0*Gamma_plus_hat))
        hyper_plane_sign = 1.0
        #print('gamma_plus is the minimum')
    else:
        
        facet_pair_idx = argwhere(-1.0*Gamma_minus_hat == min(-1.0*Gamma_minus_hat))
        hyper_plane_sign = -1.0
        #print('gamma_minus is the minimum')

    facet_pair_min = argwhere(Gamma_sum == min(Gamma_sum))
    
    #print('in whole is Gamma_plus',Gamma_plus_hat)
    #print('in whole is Gamma_minus',Gamma_minus_hat)
    
    #print('facet_pair_min',facet_pair_min)
    #print('shape pf gamma_plus',shape(Gamma_plus))
    #print('facet_pair_idx',facet_pair_idx)
    #input('facet-pair')


     ##### There are all estimated parameters
    Gamma_min_index_hat = unravel_index(argmax(-1*Gamma_total_hat,axis=None),Gamma_total.shape)




    #input('wait here')

    # eq.33 is here - Continuous analytical gamma_minimum
    Gamma_min_softmax = -1*robot_functions.smooth_max(-1*Gamma_total_hat*10000)/10000.0  # Previsois value was 10000

    #print('Gamma_min',Gamma_min)
    #print('Gamma_min_index_hat',Gamma_min_index_hat)
    #print('facet_pair_idx',facet_pair_idx)
    





    return Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign
