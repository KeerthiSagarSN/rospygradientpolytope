# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:24:34 2022

@author: keerthi.sagar
"""


try:
    import polytope as pc
except ImportError:
    raise ImportError('Polytope library not installed - pip install polytope')

#import polytope as pc
from logging import raiseExceptions
from socket import close
from scipy.spatial import ConvexHull
from pypoman import compute_polytope_vertices, compute_chebyshev_center
import rospy




#from pycapacity import velocity_polytope_withfaces
from numpy import hstack, vstack, array, dot, transpose, append, shape, zeros, ones,cross, allclose, empty
from numpy.linalg import norm
from rospygradientpolytope.polytope_functions import get_polytope_hyperplane, get_cartesian_polytope_hyperplane,get_capacity_margin
from rospygradientpolytope.polytope_functions import get_constraint_joint_limit_polytope, get_constraint_obstacle_joint_polytope
from rospygradientpolytope.polytope_functions import get_constraint_polytope,get_constraint_hsm_polytope
# Plot polytope here with polytope library
import time




# Plot KUKA - Velocity polytope here
def __init__():
    print('Start Visual library')
    

def desired_polytope(cartesian_desired_vertices):

    ''' 
    
    Desired Polytope: Cartesian Force/ velocity polytope

    Input:

    Desired cartesian vertex set
       
    
    

    cartesian_desired_vertices = 0.75*array([[0.20000, 0.50000, 0.50000],
                                        [0.50000, -0.10000, 0.50000],
                                        [0.50000, 0.50000, -0.60000],
                                        [0.50000, -0.10000, -0.60000],
                                        [-0.30000, 0.50000, 0.50000],
                                        [-0.30000, -0.10000, 0.50000],
                                        [-0.30000, 0.50000, -0.60000],
                                        [-0.30000, -0.10000, -0.60000]])
    '''
    
    polytope_vertices = cartesian_desired_vertices
    hull = ConvexHull(polytope_vertices, qhull_options='Qs QJ')

    
    polytope_faces = hull.simplices
    
    return polytope_vertices, polytope_faces


## Plot 2D - Polytopes for Cartesian 
def force_polytope_2D(W,qdot_min, qdot_max,cartesian_desired_vertices,sigmoid_slope):

    from polytope_functions_2D import get_polytope_hyperplane,get_capacity_margin

    qdot_max = array(qdot_max)
    qdot_min = array(qdot_min)
    cartesian_dof_input = array([True, True, False, False, False, False])

    active_joints = shape(W)[1]

    h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
        W, active_joints, cartesian_dof_input, qdot_min, qdot_max, sigmoid_slope)


    Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx,hyper_plane_sign = \
        get_capacity_margin(W,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints,cartesian_dof_input,qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope)
    
    
    ################## Actual - Polytope #########################################################################
    
    A = None

    B_matrix = None
    A = vstack((n_k, -n_k))

    B_matrix = array([[-10000]])

    
    for i in range(len(n_k)):
        B = dot(transpose(A[i, :]), p_plus[i, :])
        B_matrix = vstack((B_matrix, B))

    for i in range(len(n_k)):
        B = dot(-transpose(A[i, :]), p_minus[i, :])
        B_matrix = vstack((B_matrix, B))
    B_matrix = B_matrix[1:, :]

    # Getting all polytope vertices here with
    # Polytope library from pypi
    # https://pypi.org/project/polytope/

    # Plot using polytope function polytope from pypi

    #print('A',A)
    #print('B_matrix',B_matrix)

    p = pc.Polytope(A, B_matrix)

    polytope_vertices = pc.extreme(p)

    # Available cartesian wrench/velocity polytope

    #print('polytope_vertices',polytope_vertices)

    hull = ConvexHull(polytope_vertices)

    polytope_faces = hull.simplices

    #print('polytope_faces',polytope_faces)
    #print('polytope_vertices',polytope_vertices)

    
    closest_vertex = cartesian_desired_vertices[facet_pair_idx[0,1]]
    ### Very bad implementation with numpy need to find a mathematical way to plot TODO

    normal_capacity = hyper_plane_sign*n_k[facet_pair_idx[0,0]]
    
    ## Assumption that traingulated faces
    capacity_margin_faces = zeros([1,2])
    
    capacity_proj_vertex = closest_vertex
    minimum_dist = 1e8
    for i in range(len(polytope_faces)):

        normal_plane = (polytope_vertices[polytope_faces[i,0]] - polytope_vertices[polytope_faces[i,1]])
            
        norm_normal_plane = norm(normal_plane)

        norm_normal_plane = normal_plane

        '''
        if norm_normal_plane != 0:
            normal_plane = normal_plane/norm_normal_plane
        else:
            raiseExceptions('Division by zero not possible')
            normal_plane = normal_plane
        '''
        if allclose(normal_capacity,normal_plane):
            point_plane_dist = (dot((polytope_vertices[polytope_faces[i,0]] - closest_vertex),normal_plane))
            
            

            if point_plane_dist > 0:
                #minimum_dist = point_plane_dist
                #input('stp here')
                minimum_dist = point_plane_dist
                capacity_margin_faces = vstack((capacity_margin_faces,polytope_faces[i,:]))
                capacity_proj_vertex = closest_vertex + normal_plane*point_plane_dist

    capacity_margin_faces = capacity_margin_faces[1:,:]

    #print('capacity_margin_faces',capacity_margin_faces)
    #capacity_margin_vertices = None
    #capacity_margin_faces = None
    
    hull.close()

    ################## Estimated - Polytope #########################################################################

    A = None

    A = vstack((n_k, -n_k))

    B_matrx = None
    B_matrix = array([[-10000]])

    
    for i in range(len(n_k)):
        B = dot(transpose(A[i, :]), p_plus_hat[i, :])
        B_matrix = vstack((B_matrix, B))

    for i in range(len(n_k)):
        B = dot(-transpose(A[i, :]), p_minus_hat[i, :])
        B_matrix = vstack((B_matrix, B))
    B_matrix = B_matrix[1:, :]

    # Getting all polytope vertices here with
    # Polytope library from pypi
    # https://pypi.org/project/polytope/

    # Plot using polytope function polytope from pypi

    p_est = pc.Polytope(A, B_matrix)

    polytope_vertices_est = pc.extreme(p_est)

    # Available cartesian wrench/velocity polytope

    hull2 = ConvexHull(polytope_vertices_est)

    polytope_faces_est = hull2.simplices

    
    closest_vertex = cartesian_desired_vertices[facet_pair_idx[0,1]]
    ### Very bad implementation with numpy need to find a mathematical way to plot TODO

    normal_capacity = hyper_plane_sign*n_k[facet_pair_idx[0,0]]
    
    ## Assumption that traingulated faces
    capacity_margin_faces_est = zeros([1,2])
    

    minimum_dist_est = 1e8
    capacity_proj_vertex_est = closest_vertex
    for i in range(len(polytope_faces_est)):

        normal_plane_est = (polytope_vertices_est[polytope_faces_est[i,0]] - polytope_vertices_est[polytope_faces_est[i,1]])
            
        norm_normal_plane_est = norm(normal_plane_est)

        norm_normal_plane_est = normal_plane_est

        '''
        if norm_normal_plane != 0:
            normal_plane_est = normal_plane_est/norm_normal_plane_est
        else:
            raiseExceptions('Division by zero not possible')
            
            normal_plane_est = normal_plane_est
        '''
        if allclose(normal_capacity,normal_plane_est):
            point_plane_dist = (dot((polytope_vertices_est[polytope_faces_est[i,0]] - closest_vertex),normal_plane_est))
            #

            if point_plane_dist > 0:
                #minimum_dist = point_plane_dist
                minimum_dist_est = point_plane_dist
                #print('point_plane_dist',point_plane_dist)
                #input('stop here') 
                capacity_margin_faces_est = vstack((capacity_margin_faces_est,polytope_faces_est[i,:]))
                capacity_proj_vertex_est = closest_vertex + normal_plane_est*point_plane_dist
                

    capacity_margin_faces_est = capacity_margin_faces_est[1:,:]

    #print('capacity_margin_faces',capacity_margin_faces)
    #capacity_margin_vertices = None
    #capacity_margin_faces = None
    
    hull2.close()

    

    
    return polytope_vertices, polytope_faces, facet_pair_idx, capacity_margin_faces, \
            capacity_proj_vertex, polytope_vertices_est, polytope_faces_est, capacity_margin_faces_est, capacity_proj_vertex_est


def velocity_polytope(JE, qdot_max, qdot_min,cartesian_desired_vertices):

    qdot_max = array(qdot_max)
    qdot_min = -1*array(qdot_max)
    active_joints = shape(JE)[1]
    cartesian_dof_input = array([True, True, True, False, False, False])

    '''
    cartesian_desired_vertices = 3*array([[0.20000, 0.50000, 0.50000],
                                        [0.50000, -0.10000, 0.50000],
                                        [0.50000, 0.50000, -0.60000],
                                        [0.50000, -0.10000, -0.60000],
                                        [-0.30000, 0.50000, 0.50000],
                                        [-0.30000, -0.10000, 0.50000],
                                        [-0.30000, 0.50000, -0.60000],
                                        [-0.30000, -0.10000, -0.60000]])
    '''
    sigmoid_slope = 100
    h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
        JE, active_joints, cartesian_dof_input, qdot_min, qdot_max, cartesian_desired_vertices, sigmoid_slope)


    Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx,hyper_plane_sign = \
        get_capacity_margin(JE,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints,cartesian_dof_input,qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope)
    
    
    ################## Actual - Polytope #########################################################################
    
    A = None

    B_matrix = None
    A = vstack((n_k, -n_k))

    B_matrix = array([[-10000]])

    
    for i in range(len(n_k)):
        B = dot(transpose(A[i, :]), p_plus[i, :])
        B_matrix = vstack((B_matrix, B))

    for i in range(len(n_k)):
        B = dot(-transpose(A[i, :]), p_minus[i, :])
        B_matrix = vstack((B_matrix, B))
    B_matrix = B_matrix[1:, :]

    # Getting all polytope vertices here with
    # Polytope library from pypi
    # https://pypi.org/project/polytope/

    # Plot using polytope function polytope from pypi

    p = pc.Polytope(A, B_matrix)

    polytope_vertices = pc.extreme(p)

    # Available cartesian wrench/velocity polytope

    hull = ConvexHull(polytope_vertices)

    polytope_faces = hull.simplices

    
    closest_vertex = cartesian_desired_vertices[facet_pair_idx[0,1]]
    ### Very bad implementation with numpy need to find a mathematical way to plot TODO

    normal_capacity = hyper_plane_sign*n_k[facet_pair_idx[0,0]]
    
    ## Assumption that traingulated faces
    capacity_margin_faces = zeros([1,3])
    
    capacity_proj_vertex = closest_vertex
    for i in range(len(polytope_faces)):

        normal_plane = cross((polytope_vertices[polytope_faces[i,0]] - polytope_vertices[polytope_faces[i,1]])\
            , (polytope_vertices[polytope_faces[i,0]] - polytope_vertices[polytope_faces[i,2]]))
        norm_normal_plane = norm(normal_plane)

        if norm_normal_plane != 0:
            normal_plane = normal_plane/norm_normal_plane
        else:
            raiseExceptions('Division by zero not possible')
            normal_plane = normal_plane
        if allclose(normal_capacity,normal_plane):
            point_plane_dist = (dot((polytope_vertices[polytope_faces[i,0]]  - closest_vertex),normal_plane)) 
            if point_plane_dist > 0:
                #minimum_dist = point_plane_dist
                
                capacity_margin_faces = vstack((capacity_margin_faces,polytope_faces[i,:]))
                capacity_proj_vertex = closest_vertex + normal_plane*point_plane_dist

    capacity_margin_faces = capacity_margin_faces[1:,:]

    #print('capacity_margin_faces',capacity_margin_faces)
    #capacity_margin_vertices = None
    #capacity_margin_faces = None
    
    hull.close()


    

    
    return polytope_vertices, polytope_faces, facet_pair_idx, capacity_margin_faces, capacity_proj_vertex

def cartesian_velocity_polytope(JE, qdot_max, qdot_min):

    #qdot_max = array(qdot_max)
    #qdot_min = -1*array(qdot_max)
    active_joints = shape(JE)[1]
    #cartesian_dof_input = array([True, True, True, False, False, False])

    p_plus,p_minus,n_k = get_cartesian_polytope_hyperplane(
        JE, active_joints, qdot_min, qdot_max)



    #print('Gamma_min_softmax',Gamma_min_softmax)
    
    ################## Actual - Polytope #########################################################################
    
    #A = None

    #B_matrix = None
    A = vstack((n_k, -n_k))

    #B_matrix = array([[-10000]])

    B_matrix = empty((shape(A)[0]),dtype=float)

    p_matrix = vstack((p_plus,p_minus))

    
    for i in range(2*len(n_k)):
        #B = dot(transpose(A[i, :]), p_plus[i, :])
        #B_matrix = vstack((B_matrix, B))
        B_matrix[i] = dot(transpose(A[i, :]), p_matrix[i, :])

    # for i in range(len(n_k),2*len(n_k)):
    #     #B = dot(-transpose(A[i, :]), p_minus[i, :])
    #     #B_matrix = vstack((B_matrix, B))
    #     B_matrix[i] = dot(transpose(A[i, :]), p_matrix[i, :])
    #B_matrix = B_matrix[1:, :]

    # Getting all polytope vertices here with
    # Polytope library from pypi
    # https://pypi.org/project/polytope/

    # Plot using polytope function polytope from pypi
    time_begin = time.time_ns() 
    #p = pc.Polytope(A, B_matrix)

    #polytope_vertices = pc.extreme(p)

    polytope_vertices_arr = compute_polytope_vertices(A, B_matrix)
    print('polytope_vertices_arr',polytope_vertices_arr)


    polytope_vertices = array(polytope_vertices_arr)
    #print('polytope_vertices',polytope_vertices)
    time_end = time.time_ns()
    #print('time_end',time_end)
    duration3 = time_end - time_begin
    print('duration3',duration3)
    # Available cartesian wrench/velocity polytope

    hull = ConvexHull(polytope_vertices)

    polytope_faces = hull.simplices


    hull.close()



    ################## Estimated - Polytope #########################################################################




    
    return polytope_vertices, polytope_faces
   

def cartesian_velocity_with_joint_limit_polytope(JE,q,qdot_min,qdot_max,q_min,q_max,q_mean,psi_max, psi_min):
    from numpy import matmul,size,shape

    #qdot_max = array(qdot_max)
    #qdot_min = -1*array(qdot_max)
    active_joints = shape(JE)[1]
    #cartesian_dof_input = array([True, True, True, False, False, False])

    A_jpl, B_jpl = get_constraint_joint_limit_polytope(
        JE,q,active_joints,qdot_min,qdot_max,q_min,q_max,q_mean,psi_max, psi_min)



    #print('Gamma_min_softmax',Gamma_min_softmax)
    
    ################## Actual - Polytope #########################################################################
    
    #A = None
    #print('A_jpl',A_jpl)
    #print('B_jpl',B_jpl)
    
    polytope_vertices_arr = compute_polytope_vertices(A_jpl, B_jpl)
    polytope_vertices = array(polytope_vertices_arr)
    #print('polytope_vertices',polytope_vertices)
    #print('shape of polytope vertices are',shape(polytope_vertices))
    J_Hess = JE[0:3,:]

    print('polytope_vertices',shape(polytope_vertices))
    
    #print('transpose(J_Hess)',transpose(J_Hess))
    polytope_vertices_cartesian = matmul(polytope_vertices,transpose(J_Hess))
    
    #polytope_vertices = matmul(JE, transpose(polytope_vertices_arr))
    # Available cartesian wrench/velocity polytope
    

    hull = ConvexHull(polytope_vertices_cartesian)

    polytope_faces = hull.simplices


    hull.close()

    
    return polytope_vertices_cartesian, polytope_faces




def cartesian_cmp_polytope(JE,q,qdot_min,qdot_max,q_min,q_max,q_mean,psi_max, psi_min,obstacle_link_vector,danger_vector,danger_parameter ):
    from numpy import matmul,size,shape,hstack, zeros, mean

    from copy import deepcopy
    

    #qdot_max = array(qdot_max)
    #qdot_min = -1*array(qdot_max)
    active_joints = shape(JE)[1]
    polytope_center = zeros(3)
    polytope_center_max = zeros(3)
    #cartesian_dof_input = array([True, True, True, False, False, False])

    A_cmp, B_cmp = get_constraint_polytope(
        JE,q,active_joints,qdot_min,qdot_max,q_min,q_max,q_mean,psi_max, psi_min, obstacle_link_vector,danger_vector,danger_parameter)

    #print('Gamma_min_softmax',Gamma_min_softmax)    
    ################## Actual - Polytope #########################################################################
    #A = None
    #print('A_jpl',A_jpl)
    #print('B_jpl',B_jpl)

    
    # print('A_cmp',A_cmp)
    # print('B_cmp',B_cmp)
    # Create a pycddlib matrix from the inequalities
    #M = hstack((B_cmp,-A_cmp))
    #polytope_matrix = cdd.Matrix(M, number_type="fraction")

    #Make it into a polytype object that pycddlib needs.
    #poly = cdd.Polyhedron(polytope_matrix)

    # Compute the vertices of the polyhedron
    #polytope_vertices = matrix.get_inequalities()
    try:
        polytope_vertices = array(compute_polytope_vertices(1.0*(A_cmp), 1.0*(B_cmp)))
        polytope_vertices = polytope_vertices
        
    except:
        polytope_vertices = array([])
    
    
    
    #polytope_vertices = array([])
    
    #polytope_tulip = pc.Polytope(A_cmp,B_cmp)

    #polytope_vertices = pc.extreme(polytope_tulip)

    #print('polytope_vertices',polytope_vertices)
    #print('shape of polytope vertices are',shape(polytope_vertices))
    J_Hess = JE[0:3,:]
    
    print('polytope_vertices',shape(polytope_vertices))
    # rows_verts = shape(polytope_vertices)
    # rows_verts_half = int(0.5*rows_verts[0])
    # print('row_vers',rows_verts)

    if polytope_vertices.any():
        polytope_vertices_cartesian = matmul(polytope_vertices,transpose(J_Hess))
        try:
            hull = ConvexHull(polytope_vertices_cartesian)
            polytope_faces = hull.simplices
            #Get centoid
            polytope_center[0] = mean(hull.points[hull.vertices,0])
            polytope_center_max[0] = max(hull.points[hull.vertices,0])
            polytope_center[1] = mean(hull.points[hull.vertices,1])
            polytope_center_max[1] = max(hull.points[hull.vertices,1])
            polytope_center[2] = mean(hull.points[hull.vertices,2])
            polytope_center_max[2] = max(hull.points[hull.vertices,2])
            #input('stop now 2')
            hull.close()
        except:
            polytope_vertices_cartesian = array([])
            
            polytope_faces = array([])

        


    else:
        
        
        polytope_vertices_cartesian = array([])
            
        polytope_faces = array([])

    
    #ellipsoid_center = compute_chebyshev_center(100.0*A_cmp, 100.0*B_cmp)
    # #ellipsoid_radius = compute_chebyshev_radius(A_cmp, B_cmp)

    # polytope_vertices_lower_cartesian = matmul(polytope_vertices[rows_verts_half:,:],transpose(J_Hess))
    # hull2 = ConvexHull(polytope_vertices_lower_cartesian)

    # polytope_lower_faces = hull2.simplices

    # ellipsoid_center = polytope_tulip.chebXc
    # ellipsoid_ball = polytope_tulip.chebR



    # , polytope_vertices_lower_cartesian, polytope_lower_faces#, ellipsoid_center, ellipsoid_ball
    return polytope_vertices_cartesian, polytope_faces, polytope_center, polytope_center_max


def cartesian_cmp_hsm_polytope(JE,q,qdot_min,qdot_max,q_min,q_max,q_mean,psi_max, psi_min,obstacle_link_vector,danger_vector,danger_parameter ):
    from numpy import matmul,size,shape,hstack, zeros, mean

    from copy import deepcopy
    

    #qdot_max = array(qdot_max)
    #qdot_min = -1*array(qdot_max)
    active_joints = shape(JE)[1]
    polytope_center = zeros(3)
    polytope_center_max = zeros(3)
    #cartesian_dof_input = array([True, True, True, False, False, False])

    A_cmp, B_cmp = get_constraint_hsm_polytope(
        JE,q,active_joints,qdot_min,qdot_max,q_min,q_max,q_mean,psi_max, psi_min, obstacle_link_vector,danger_vector,danger_parameter)

    #print('Gamma_min_softmax',Gamma_min_softmax)    
    ################## Actual - Polytope #########################################################################
    #A = None
    #print('A_jpl',A_jpl)
    #print('B_jpl',B_jpl)

    
    # print('A_cmp',A_cmp)
    # print('B_cmp',B_cmp)
    # Create a pycddlib matrix from the inequalities
    #M = hstack((B_cmp,-A_cmp))
    #polytope_matrix = cdd.Matrix(M, number_type="fraction")

    #Make it into a polytype object that pycddlib needs.
    #poly = cdd.Polyhedron(polytope_matrix)

    # Compute the vertices of the polyhedron
    #polytope_vertices = matrix.get_inequalities()
    try:
        polytope_vertices = array(compute_polytope_vertices(100.0*(A_cmp), 100.0*(B_cmp)))
        polytope_vertices = polytope_vertices*(100**(-1))
        
    except:
        polytope_vertices = array([])
    
    
    
    #polytope_vertices = array([])
    
    #polytope_tulip = pc.Polytope(A_cmp,B_cmp)

    #polytope_vertices = pc.extreme(polytope_tulip)

    #print('polytope_vertices',polytope_vertices)
    #print('shape of polytope vertices are',shape(polytope_vertices))
    J_Hess = JE[0:3,:]
    
    print('polytope_vertices',shape(polytope_vertices))
    # rows_verts = shape(polytope_vertices)
    # rows_verts_half = int(0.5*rows_verts[0])
    # print('row_vers',rows_verts)

    if polytope_vertices.any():
        polytope_vertices_cartesian = matmul(polytope_vertices,transpose(J_Hess))
        try:
            hull = ConvexHull(polytope_vertices_cartesian)
            polytope_faces = hull.simplices
            #Get centoid
            polytope_center[0] = mean(hull.points[hull.vertices,0])
            polytope_center_max[0] = max(hull.points[hull.vertices,0])
            polytope_center[1] = mean(hull.points[hull.vertices,1])
            polytope_center_max[1] = max(hull.points[hull.vertices,1])
            polytope_center[2] = mean(hull.points[hull.vertices,2])
            polytope_center_max[2] = max(hull.points[hull.vertices,2])
            #input('stop now 2')
            hull.close()
        except:
            polytope_vertices_cartesian = array([])
            
            polytope_faces = array([])

        


    else:
        
        
        polytope_vertices_cartesian = array([])
            
        polytope_faces = array([])

    
    #ellipsoid_center = compute_chebyshev_center(100.0*A_cmp, 100.0*B_cmp)
    # #ellipsoid_radius = compute_chebyshev_radius(A_cmp, B_cmp)

    # polytope_vertices_lower_cartesian = matmul(polytope_vertices[rows_verts_half:,:],transpose(J_Hess))
    # hull2 = ConvexHull(polytope_vertices_lower_cartesian)

    # polytope_lower_faces = hull2.simplices

    # ellipsoid_center = polytope_tulip.chebXc
    # ellipsoid_ball = polytope_tulip.chebR



    # , polytope_vertices_lower_cartesian, polytope_lower_faces#, ellipsoid_center, ellipsoid_ball
    return polytope_vertices_cartesian, polytope_faces, polytope_center, polytope_center_max


def cartesian_cmp_polytope(JE,q,qdot_min,qdot_max,q_min,q_max,q_mean,psi_max, psi_min,obstacle_link_vector,danger_vector,danger_parameter ):
    from numpy import matmul,size,shape,hstack, zeros, mean

    from copy import deepcopy
    

    #qdot_max = array(qdot_max)
    #qdot_min = -1*array(qdot_max)
    active_joints = shape(JE)[1]
    polytope_center = zeros(3)
    polytope_center_max = zeros(3)
    #cartesian_dof_input = array([True, True, True, False, False, False])

    A_cmp, B_cmp = get_constraint_polytope(
        JE,q,active_joints,qdot_min,qdot_max,q_min,q_max,q_mean,psi_max, psi_min, obstacle_link_vector,danger_vector,danger_parameter)

    #print('Gamma_min_softmax',Gamma_min_softmax)    
    ################## Actual - Polytope #########################################################################
    #A = None
    #print('A_jpl',A_jpl)
    #print('B_jpl',B_jpl)

    
    # print('A_cmp',A_cmp)
    # print('B_cmp',B_cmp)
    # Create a pycddlib matrix from the inequalities
    #M = hstack((B_cmp,-A_cmp))
    #polytope_matrix = cdd.Matrix(M, number_type="fraction")

    #Make it into a polytype object that pycddlib needs.
    #poly = cdd.Polyhedron(polytope_matrix)

    # Compute the vertices of the polyhedron
    #polytope_vertices = matrix.get_inequalities()
    try:
        polytope_vertices = array(compute_polytope_vertices(100.0*(A_cmp), 100.0*(B_cmp)))
        polytope_vertices = polytope_vertices*(100**(-1))
        
    except:
        polytope_vertices = array([])
    
    
    
    #polytope_vertices = array([])
    
    #polytope_tulip = pc.Polytope(A_cmp,B_cmp)

    #polytope_vertices = pc.extreme(polytope_tulip)

    #print('polytope_vertices',polytope_vertices)
    #print('shape of polytope vertices are',shape(polytope_vertices))
    J_Hess = JE[0:3,:]
    
    print('polytope_vertices',shape(polytope_vertices))
    # rows_verts = shape(polytope_vertices)
    # rows_verts_half = int(0.5*rows_verts[0])
    # print('row_vers',rows_verts)

    if polytope_vertices.any():
        polytope_vertices_cartesian = matmul(polytope_vertices,transpose(J_Hess))
        try:
            hull = ConvexHull(polytope_vertices_cartesian)
            polytope_faces = hull.simplices
            #Get centoid
            polytope_center[0] = mean(hull.points[hull.vertices,0])
            polytope_center_max[0] = max(hull.points[hull.vertices,0])
            polytope_center[1] = mean(hull.points[hull.vertices,1])
            polytope_center_max[1] = max(hull.points[hull.vertices,1])
            polytope_center[2] = mean(hull.points[hull.vertices,2])
            polytope_center_max[2] = max(hull.points[hull.vertices,2])
            #input('stop now 2')
            hull.close()
        except:
            polytope_vertices_cartesian = array([])
            
            polytope_faces = array([])

        


    else:
        
        
        polytope_vertices_cartesian = array([])
            
        polytope_faces = array([])

    
    #ellipsoid_center = compute_chebyshev_center(100.0*A_cmp, 100.0*B_cmp)
    # #ellipsoid_radius = compute_chebyshev_radius(A_cmp, B_cmp)

    # polytope_vertices_lower_cartesian = matmul(polytope_vertices[rows_verts_half:,:],transpose(J_Hess))
    # hull2 = ConvexHull(polytope_vertices_lower_cartesian)

    # polytope_lower_faces = hull2.simplices

    # ellipsoid_center = polytope_tulip.chebXc
    # ellipsoid_ball = polytope_tulip.chebR



    # , polytope_vertices_lower_cartesian, polytope_lower_faces#, ellipsoid_center, ellipsoid_ball
    return polytope_vertices_cartesian, polytope_faces, polytope_center, polytope_center_max



def cartesian_velocity_with_obstacle_polytope(JE,obstacle_link_vector,danger_distance):
    from numpy import matmul,size,shape,identity, vstack
    from copy import deepcopy
    from numpy.linalg import pinv

    #qdot_max = array(qdot_max)
    #qdot_min = -1*array(qdot_max)
    active_joints = shape(JE)[1]
    #cartesian_dof_input = array([True, True, True, False, False, False])

    A_obs, B_obs = get_constraint_obstacle_joint_polytope(JE, obstacle_link_vector, danger_distance)



    #print('Gamma_min_softmax',Gamma_min_softmax)
    
    ################## Actual - Polytope #########################################################################
    
    #A = None
    #print('A_jpl',A_jpl)
    #print('B_jpl',B_jpl)
    
    # I_obs = identity(12)


    # A_obs = vstack((A_obs,I_obs))
    # B_obs_zero = deepcopy(B_obs)
    # B_obs_zero[:] = 100
    # B_obs = hstack((array([B_obs]),array([B_obs_zero])))
    A_obs = transpose(A_obs)
    B_obs = transpose(array([B_obs]))
    print('A_obs', A_obs)
    print('B_obs',B_obs)
    print('shape of B_obs',shape(B_obs))
    #input('stop inside pypoman')
    #A_obs = identity(active_joints)
    
    polytope_vertices_arr = compute_polytope_vertices(A_obs, B_obs)
    print('polytope_vertices',polytope_vertices)
    #polytope_vertices_arr = pc.Polytope(transpose(A_obs), B_obs)
    polytope_vertices = array(polytope_vertices_arr)
    #print('polytope_vertices',polytope_vertices)
    #print('shape of polytope vertices are',shape(polytope_vertices))
    J_Hess = JE[0:3,:]

    print('polytope_vertices',shape(polytope_vertices))
    print('polytope_vertices',polytope_vertices)
    
    #print('transpose(J_Hess)',transpose(J_Hess))
    polytope_vertices_cartesian = matmul(J_Hess,transpose(polytope_vertices))


    print('polytope__cartesian',polytope_vertices_cartesian)
    #input('stop now')
    #polytope_vertices = matmul(JE, transpose(polytope_vertices_arr))
    # Available cartesian wrench/velocity polytope
    

    hull = ConvexHull(polytope_vertices_cartesian)

    polytope_faces = hull.simplices


    hull.close()

    
    return polytope_vertices_cartesian, polytope_faces

def velocity_polytope_with_estimation(JE, qdot_max, qdot_min,cartesian_desired_vertices, sigmoid_slope):

    qdot_max = array(qdot_max)
    qdot_min = -1*array(qdot_max)
    active_joints = shape(JE)[1]
    cartesian_dof_input = array([True, True, True, False, False, False])

    '''
    cartesian_desired_vertices =  0.1*array([[0.20000, 0.50000, 0.50000],
                                        [0.50000, -0.10000, 0.50000],
                                        [0.50000, 0.50000, -0.60000],
                                        [0.50000, -0.10000, -0.60000],
                                        [-0.30000, 0.50000, 0.50000],
                                        [-0.30000, -0.10000, 0.50000],
                                        [-0.30000, 0.50000, -0.60000],
                                        [-0.30000, -0.10000, -0.60000]])
    '''    
    #sigmoid_slope = 150
    h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
        JE, active_joints, cartesian_dof_input, qdot_min, qdot_max, cartesian_desired_vertices, sigmoid_slope)


    Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx,hyper_plane_sign = \
        get_capacity_margin(JE,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                        active_joints,cartesian_dof_input,qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope)
    #print('Gamma_min_softmax',Gamma_min_softmax)
    
    ################## Actual - Polytope #########################################################################
    
    A = None

    B_matrix = None
    A = vstack((n_k, -n_k))

    B_matrix = array([[-10000]])

    
    for i in range(len(n_k)):
        B = dot(transpose(A[i, :]), p_plus[i, :])
        B_matrix = vstack((B_matrix, B))

    for i in range(len(n_k)):
        B = dot(-transpose(A[i, :]), p_minus[i, :])
        B_matrix = vstack((B_matrix, B))
    B_matrix = B_matrix[1:, :]

    # Getting all polytope vertices here with
    # Polytope library from pypi
    # https://pypi.org/project/polytope/

    # Plot using polytope function polytope from pypi

    p = pc.Polytope(A, B_matrix)

    polytope_vertices = pc.extreme(p)

    # Available cartesian wrench/velocity polytope

    hull = ConvexHull(polytope_vertices)

    polytope_faces = hull.simplices

    
    closest_vertex = cartesian_desired_vertices[facet_pair_idx[0,1]]
    ### Very bad implementation with numpy need to find a mathematical way to plot TODO

    normal_capacity = hyper_plane_sign*n_k[facet_pair_idx[0,0]]
    
    ## Assumption that traingulated faces
    capacity_margin_faces = zeros([1,3])
    
    capacity_proj_vertex = closest_vertex
    minimum_dist = 1e8
    for i in range(len(polytope_faces)):

        normal_plane = cross((polytope_vertices[polytope_faces[i,0]] - polytope_vertices[polytope_faces[i,1]])\
            , (polytope_vertices[polytope_faces[i,0]] - polytope_vertices[polytope_faces[i,2]]))
        norm_normal_plane = norm(normal_plane)

        if norm_normal_plane != 0:
            normal_plane = normal_plane/norm_normal_plane
        else:
            #raiseExceptions('Division by zero not possible')
            print('Division by zero not possible')
            normal_plane = normal_plane
        if allclose(normal_capacity,normal_plane):
            point_plane_dist = (dot((polytope_vertices[polytope_faces[i,0]] - closest_vertex),normal_plane))
            
            

            if point_plane_dist > 0:
                #minimum_dist = point_plane_dist
                #input('stp here')
                minimum_dist = point_plane_dist
                capacity_margin_faces = vstack((capacity_margin_faces,polytope_faces[i,:]))
                capacity_proj_vertex = closest_vertex + normal_plane*point_plane_dist

    capacity_margin_faces = capacity_margin_faces[1:,:]

    #print('capacity_margin_faces',capacity_margin_faces)
    #capacity_margin_vertices = None
    #capacity_margin_faces = None
    
    hull.close()

    ################## Estimated - Polytope #########################################################################

    A = None

    A = vstack((n_k, -n_k))

    B_matrx = None
    B_matrix = array([[-10000]])

    
    for i in range(len(n_k)):
        B = dot(transpose(A[i, :]), p_plus_hat[i, :])
        B_matrix = vstack((B_matrix, B))

    for i in range(len(n_k)):
        B = dot(-transpose(A[i, :]), p_minus_hat[i, :])
        B_matrix = vstack((B_matrix, B))
    B_matrix = B_matrix[1:, :]

    # Getting all polytope vertices here with
    # Polytope library from pypi
    # https://pypi.org/project/polytope/

    # Plot using polytope function polytope from pypi

    p_est = pc.Polytope(A, B_matrix)

    polytope_vertices_est = pc.extreme(p_est)

    # Available cartesian wrench/velocity polytope

    hull2 = ConvexHull(polytope_vertices_est)

    polytope_faces_est = hull2.simplices

    
    closest_vertex = cartesian_desired_vertices[facet_pair_idx[0,1]]
    ### Very bad implementation with numpy need to find a mathematical way to plot TODO

    normal_capacity = hyper_plane_sign*n_k[facet_pair_idx[0,0]]
    
    ## Assumption that traingulated faces
    capacity_margin_faces_est = zeros([1,3])
    

    minimum_dist_est = 1e8
    capacity_proj_vertex_est = closest_vertex
    for i in range(len(polytope_faces_est)):

        normal_plane_est = cross((polytope_vertices_est[polytope_faces_est[i,0]] - polytope_vertices_est[polytope_faces_est[i,1]])\
            , (polytope_vertices_est[polytope_faces_est[i,0]] - polytope_vertices_est[polytope_faces_est[i,2]]))
        norm_normal_plane_est = norm(normal_plane_est)

        if norm_normal_plane != 0:
            normal_plane_est = normal_plane_est/norm_normal_plane_est
        else:
            #raiseExceptions('Division by zero not possible')
            print('Division by zero not possible')
            
            normal_plane_est = normal_plane_est
        if allclose(normal_capacity,normal_plane_est):
            point_plane_dist = (dot((polytope_vertices_est[polytope_faces_est[i,0]] - closest_vertex),normal_plane_est))
            #

            if point_plane_dist > 0:
                #minimum_dist = point_plane_dist
                minimum_dist_est = point_plane_dist
                #print('point_plane_dist',point_plane_dist)
                #input('stop here') 
                capacity_margin_faces_est = vstack((capacity_margin_faces_est,polytope_faces_est[i,:]))
                capacity_proj_vertex_est = closest_vertex + normal_plane_est*point_plane_dist
                

    capacity_margin_faces_est = capacity_margin_faces_est[1:,:]

    #print('capacity_margin_faces',capacity_margin_faces)
    #capacity_margin_vertices = None
    #capacity_margin_faces = None
    
    hull2.close()



    
    return polytope_vertices, polytope_faces, facet_pair_idx, capacity_margin_faces, \
            capacity_proj_vertex, polytope_vertices_est, polytope_faces_est, capacity_margin_faces_est, capacity_proj_vertex_est,p,p_est,Gamma_min_softmax
            
def pycapacity_polytope(JE, qdot_max, qdot_min):
    '''
    ### Get pycapacity polytope
    Args

    JE - jacobian - n- DOF

    qdot_max - maximum joint velocity/toques

    qdot_min - minimum joint velocity/torque



    ## Dependent on external library

    pycapacity - pip install pycapacity




    '''
    try:
        from pycapacity.robot import velocity_polytope_withfaces
    except ImportError:
        raise ImportError('pycapacity not installed - pip install pycapacity')

    qdot_max = array(qdot_max)
    qdot_min = array(qdot_min)

    polytope_verts, polytope_faces = velocity_polytope_withfaces(
        JE[0:3, :], qdot_max, qdot_min)

    return polytope_verts, polytope_faces


def plot_polytope_actual(ax, JE, active_joints, cartesian_dof_input, qdot_min, qdot_max, cartesian_desired_vertices, sigmoid_slope):
    '''
    Get available cartesian polytope and desired polytope

    NO external library dependency


    Arguments

    Input
    --------------

    JE - n-DOF jacobian  [v w] = JE.q

    active-joints - k- dof active joints

    cartesian_dof_input = array([True,True,True,False,False,False]) #### For translational

    ---------------



    '''
    v_k_d = cartesian_desired_vertices
    ax.cla()
    h_plus, h_plus_hat, h_minus, h_minus_hat, p_plus, p_minus, n_k, Nmatrix, Nnot = get_polytope_hyperplane(
        JE, active_joints, cartesian_dof_input, qdot_min, qdot_max, cartesian_desired_vertices, sigmoid_slope)
    A = vstack((n_k, -n_k))

    B_matrix = array([[-10000]])

    p_matrix = vstack((p_plus, p_minus))
    for i in range(len(n_k)):
        B = dot(transpose(A[i, :]), p_plus[i, :])
        B_matrix = vstack((B_matrix, B))

    for i in range(len(n_k)):
        B = dot(-transpose(A[i, :]), p_minus[i, :])
        B_matrix = vstack((B_matrix, B))
    B_matrix = B_matrix[1:, :]

    # Getting all polytope vertices here with
    # Polytope library from pypi
    # https://pypi.org/project/polytope/

    # Plot using polytope function polytope from pypi

    p = pc.Polytope(A, B_matrix)

    polytope_vertices = pc.extreme(p)

    # I commented below the poltyope

    #hull1 = ConvexHull()
    # First find the convex hull
    # Available cartesian wrench/velocity polytope

    hull2 = ConvexHull(polytope_vertices, qhull_options='Qs QJ')
    
    #input('wait here')
    # if plot_polytope_on == True:

    for s in hull2.simplices:
        s = append(s, s[0])  # Here we cycle back to the first coordinate
        #canvas_polytope.ax.plot(polytope_vertices[s, 0]+self.robot.end_effector_position[0], polytope_vertices[s, 1]+self.robot.end_effector_position[1], polytope_vertices[s, 2]+self.robot.end_effector_position[2], "k-")
        ax.plot(
            polytope_vertices[s, 0], polytope_vertices[s, 1], polytope_vertices[s, 2], "k-")
        #canvas_polytope.ax.scatter3D(self.robot.end_effector_position[0],self.robot.end_effector_position[1],self.robot.end_effector_position[2],c= 'k',s=10)
    #canvas_polytope.ax.plot(polytope_vertices[s, 0], polytope_vertices[s, 1], polytope_vertices[s, 2], "y-")

    ax.scatter3D(v_k_d[:, 0], v_k_d[:, 1], v_k_d[:, 2], 'cyan')
    verts = [list(zip(v_k_d[:, 0], v_k_d[:, 1], v_k_d[:, 2]))]

    #verts = hs.intersections
    hull = ConvexHull(v_k_d, qhull_options='Qs QJ')

    print('hull.simplices', hull.simplices)
    print('len(hull.simplices)', len(hull.simplices))

    polytope_color = "g"

    for s in hull.simplices:
        s = append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(v_k_d[s, 0], v_k_d[s, 1], v_k_d[s, 2], color=polytope_color)

    # canvas_polytope.generate_axis()
    ax.plt.pause(0.01)
    ax.canvas.draw_idle()
    return ax


def plot_polytope_estimated(ax, JE, active_joints, cartesian_dof_input, qdot_min, qdot_max, cartesian_desired_vertices, sigmoid_slope):
    '''
    Get available cartesian polytope and desired polytope

    NO external library dependency



    '''

    try:
        import Polytope as pc
    except ImportError:
        raise ImportError(
            'Polytope library not installed - pip install Polytope')

    # Desired cartesian velocity/force polytope - Vertex set
    v_k_d = cartesian_desired_vertices

    h_plus, h_plus_hat, h_minus, h_minus_hat, p_plus, p_minus, p_plus_hat, p_minus_hat, n_k, Nmatrix, Nnot = get_polytope_hyperplane(
        JE, active_joints, cartesian_dof_input, qdot_min, qdot_max, cartesian_desired_vertices, sigmoid_slope)

    # Desired cartesian velocity/force polytope

    A = vstack((n_k, -n_k))

    B_matrix = array([[-10000]])

    p_matrix = vstack((p_plus, p_minus))
    for i in range(len(n_k)):
        B = dot(transpose(A[i, :]), p_plus[i, :])
        B_matrix = vstack((B_matrix, B))

    for i in range(len(n_k)):
        B = dot(-transpose(A[i, :]), p_minus[i, :])
        B_matrix = vstack((B_matrix, B))
    B_matrix = B_matrix[1:, :]

    # Getting all polytope vertices here with
    # Polytope library from pypi
    # https://pypi.org/project/polytope/

    # Plot using polytope function polytope from pypi

    p = pc.Polytope(A, B_matrix)

    polytope_vertices = pc.extreme(p)

    # I commented below the poltyope

    #hull1 = ConvexHull()
    # First find the convex hull
    hull2 = ConvexHull(polytope_vertices, qhull_options='Qs QJ')
    # print('hull2.simplices',hull2.simplices)
    # print('len(hull2.simplices)',len(hull2.simplices))
    #input('wait here')

    for s in hull2.simplices:
        s = append(s, s[0])  # Here we cycle back to the first coordinate
        #canvas_polytope.ax.plot(polytope_vertices[s, 0]+self.robot.end_effector_position[0], polytope_vertices[s, 1]+self.robot.end_effector_position[1], polytope_vertices[s, 2]+self.robot.end_effector_position[2], "k-")
        ax.plot(
            polytope_vertices[s, 0], polytope_vertices[s, 1], polytope_vertices[s, 2], "k-")
        #canvas_polytope.ax.scatter3D(self.robot.end_effector_position[0],self.robot.end_effector_position[1],self.robot.end_effector_position[2],c= 'k',s=10)
    #canvas_polytope.ax.plot(polytope_vertices[s, 0], polytope_vertices[s, 1], polytope_vertices[s, 2], "y-")

    ax.scatter3D(v_k_d[:, 0], v_k_d[:, 1], v_k_d[:, 2], 'cyan')
    verts = [list(zip(v_k_d[:, 0], v_k_d[:, 1], v_k_d[:, 2]))]

    #verts = hs.intersections

    # Desired cartesian velocity/force vertices
    hull = ConvexHull(v_k_d)

    print('hull.simplices', hull.simplices)
    print('len(hull.simplices)', len(hull.simplices))

    polytope_color = "g"

    for s in hull.simplices:
        s = append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(v_k_d[s, 0], v_k_d[s, 1], v_k_d[s, 2], color=polytope_color)

    # canvas_polytope.generate_axis()
    ax.pause(0.01)
    ax.cla()


def plot_polytope_capacity_margin_gradient(ax, JE, active_joints, cartesian_dof_input, qdot_min, qdot_max, cartesian_desired_vertices, sigmoid_slope):
    '''
    Get available cartesian polytope and desired polytope

    NO external library dependency



    '''

    # Desired cartesian velocity/force polytope - Vertex set
    v_k_d = cartesian_desired_vertices

    # Get all hyperplane parameters here
    h_plus, h_plus_hat, h_minus, h_minus_hat, p_plus, p_minus, p_plus_hat, p_minus_hat, n_k, Nmatrix, Nnot = get_polytope_hyperplane(
        JE, active_joints, cartesian_dof_input, qdot_min, qdot_max, cartesian_desired_vertices, sigmoid_slope)

    # Get estimated capacity margin here

    Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat = get_capacity_margin(JE, n_k, h_plus, h_plus_hat, h_minus, h_minus_hat,
                                                                                                                      active_joints, cartesian_dof_input, qdot_min, qdot_max, cartesian_desired_vertices, sigmoid_slope)

    A = vstack((n_k, -n_k))

    B_matrix = array([[-10000]])

    p_matrix = vstack((p_plus, p_minus))
    for i in range(len(n_k)):
        B = dot(transpose(A[i, :]), p_plus[i, :])
        B_matrix = vstack((B_matrix, B))

    for i in range(len(n_k)):
        B = dot(-transpose(A[i, :]), p_minus[i, :])
        B_matrix = vstack((B_matrix, B))
    B_matrix = B_matrix[1:, :]

    # Getting all polytope vertices here with
    # Polytope library from pypi
    # https://pypi.org/project/polytope/

    # Plot using polytope function polytope from pypi

    p = pc.Polytope(A, B_matrix)

    polytope_vertices = pc.extreme(p)

    # I commented below the poltyope

    #hull1 = ConvexHull()
    # First find the convex hull
    hull2 = ConvexHull(polytope_vertices, qhull_options='Qs QJ')
    # print('hull2.simplices',hull2.simplices)
    # print('len(hull2.simplices)',len(hull2.simplices))
    #input('wait here')

    # Are simplices the facets of the polytope ??
    for s in hull2.simplices:
        s = append(s, s[0])  # Here we cycle back to the first coordinate
        #canvas_polytope.ax.plot(polytope_vertices[s, 0]+self.robot.end_effector_position[0], polytope_vertices[s, 1]+self.robot.end_effector_position[1], polytope_vertices[s, 2]+self.robot.end_effector_position[2], "k-")
        ax.plot(
            polytope_vertices[s, 0], polytope_vertices[s, 1], polytope_vertices[s, 2], "k-")
        #canvas_polytope.ax.scatter3D(self.robot.end_effector_position[0],self.robot.end_effector_position[1],self.robot.end_effector_position[2],c= 'k',s=10)
    #canvas_polytope.ax.plot(polytope_vertices[s, 0], polytope_vertices[s, 1], polytope_vertices[s, 2], "y-")

    ax.scatter3D(v_k_d[:, 0], v_k_d[:, 1], v_k_d[:, 2], 'cyan')
    verts = [list(zip(v_k_d[:, 0], v_k_d[:, 1], v_k_d[:, 2]))]

    #verts = hs.intersections

    # Desired cartesian velocity/force vertices
    hull = ConvexHull(v_k_d)

    print('hull.simplices', hull.simplices)
    print('len(hull.simplices)', len(hull.simplices))

    if Gamma_min_softmax > 0:
        polytope_color = "g"
    else:
        polytope_color = "r"

    for s in hull.simplices:
        s = append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(v_k_d[s, 0], v_k_d[s, 1], v_k_d[s, 2], color=polytope_color)

    # canvas_polytope.generate_axis()
    ax.pause(0.01)
    ax.cla()
