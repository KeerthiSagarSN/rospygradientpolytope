# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:46:19 2022

@author: keerthi.sagar
"""

import polytope as pc

from numpy import shape,arange,array,zeros,cross,count_nonzero,transpose,matmul,dot
import polytope
import rospygradientpolytope.robot_functions
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import itertools
from rospygradientpolytope.linearalgebra import V_unit,check_ndarray
from numpy import unravel_index,argmax,min,hstack,vstack


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


    #print('Nmatrix,,Nnot')

    #print('Nmatrix is',self.Nmatrix)

    #print('self.Nnot is',self.Nnot)
    number_of_combinations = shape(Nmatrix)[0]

    # Generalized cross-product need to be implemented in the future
    # Here for Vector only of size - 3 is implemented in this code
    #print('self.cartesian_dof',self.cartesian_dof)
    ### Screw is represented as : $ = [v w] from the jacobian


    v_k = zeros(shape = (cartesian_dof,active_joints))


    v_k = JE[cartesian_dof_mask,:]
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

        n_k[i] = (V_unit(cross(v_k[:,Nmatrix[i,0]], v_k[:,Nmatrix[i,1]])))
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

        p_plus[i,:] = h_plus[i]*n_k[i,:]  + transpose(matmul(JE[0:3,:],qdot_min))

        p_minus[i,:] = h_minus[i]*n_k[i,:]  + transpose(matmul(JE[0:3,:],qdot_min))


    ##### Estimated parameters are here

    for i in range(len(l_k)):
        #p_plus = h_plus[i]*n[i,:] + matmul(JE,qmin)
        #p_plus = np.vstack((p_plus,h_plus[i]*n[i,:]))
        p_plus_hat[i,:] = h_plus_hat[i]*n_k[i,:]  + transpose(matmul(JE[0:3,:],qdot_min))
        #p_minus = h_minus[i]*n[i,:] + matmul(JE,qmin)
        p_minus_hat[i,:] = h_minus_hat[i]*n_k[i,:]  + transpose(matmul(JE[0:3,:],qdot_min))


        '''
        print('p_plus',self.p_plus)
        print('p_minus is',self.p_minus)
        print('p_plus_hat is',self.p_plus_hat)
        print('p_minus_hat',self.p_minus_hat)
        '''
    #input('wait here')

    return h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot





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


    v_k = JE[cartesian_dof_mask,:]
    JE = v_k

    Gamma_plus_LHS = transpose(array([h_plus])) + matmul(matmul(n_k,JE),transpose(array([qdot_min])))


    #input('wait here')
    Gamma_plus_hat_LHS = transpose(array([h_plus_hat])) + matmul(matmul(n_k,JE),transpose(array([qdot_min])))

    Gamma_minus_LHS = transpose(array([h_minus])) + matmul(matmul(n_k,JE),transpose(array([qdot_min])))

    Gamma_minus_hat_LHS = transpose(array([h_minus_hat])) + matmul(matmul(n_k,JE),transpose(array([qdot_min])))
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

        Gamma_minus[:,vertex] = transpose(Gamma_minus_LHS  + matmul(n_k, transpose((array([v_k_d[vertex,:]])))))
        #self.Gamma_plus[facets,vertex]= hstack((self.Gamma_plus,transpose(array([h_plus]))+ matmul(matmul(n, transpose(JE)), qmin) - matmul(n, transpose(desired_vertices[vertex, :]))))
        #self.Gamma_minus[facets,vertex]= hstack((Gamma_minus,transpose(array([h_minus]))+ matmul(matmul(-n, transpose(JE)), qmin) - matmul(-n, transpose(desired_vertices[vertex, :]))))
        #self.Gamma_minus[facets,vertex] = abs(matmul(transpose(self.n_k[facets,:]),self.p_minus[facets,:]) - matmul(-1*transpose(self.n_k[facets,:]),self.v_k_d[vertex,:]))

        Gamma_plus_hat[:,vertex] =  transpose(Gamma_plus_hat_LHS  - matmul(n_k, transpose((array([v_k_d[vertex,:]])))))

        ### estimated parameteres are here
        Gamma_minus_hat[:,vertex] = transpose(Gamma_minus_hat_LHS  + matmul(n_k, transpose((array([v_k_d[vertex,:]])))))



    #print('Gamma_plus is',self.Gamma_plus)
    #input('stop here')

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










     ##### There are all estimated parameters
    Gamma_min_index_hat = unravel_index(argmax(-1*Gamma_total_hat,axis=None),Gamma_total.shape)

    # print('self.Gamma_min_index_hat',self.Gamma_min_index_hat)

    ## Find the vertex and facet pair here
    Gamma_plus_min = min(Gamma_plus_hat)
    Gamma_plus_min_index = unravel_index(argmax(-1*Gamma_plus_hat),Gamma_total.shape)
    Gamma_minus_min = min(Gamma_minus_hat)

    if Gamma_plus_min < Gamma_minus_min:

        Gamma_min_index = unravel_index(argmax(-1*Gamma_plus_hat),Gamma_plus_hat.shape)
        Gamma_min_plus = True
    else:
        Gamma_min_index = unravel_index(argmax(-1*Gamma_minus_hat),Gamma_minus_hat.shape)
        Gamma_min_plus = False


    #input('wait here')

    # eq.33 is here - Continuous analytical gamma_minimum
    Gamma_min_softmax = -1*robot_functions.smooth_max(-1*Gamma_total_hat*1000)/1000.0  # Previsois value was 10000








    return Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat
