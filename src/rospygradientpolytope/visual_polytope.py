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
from scipy.spatial import ConvexHull
#from pycapacity import velocity_polytope_withfaces
from numpy import hstack, vstack, array, dot, transpose, append, shape
from rospygradientpolytope.polytope_functions import get_polytope_hyperplane, get_capacity_margin
# Plot polytope here with polytope library

# Equation 22
# A,B Matrix


# Plot KUKA - Velocity polytope here

def velocity_polytope(JE, qdot_max, qdot_min):

    qdot_max = array(qdot_max)
    qdot_min = -1*array(qdot_max)
    active_joints = shape(JE)[1]
    cartesian_dof_input = array([True, True, True, False, False, False])

    cartesian_desired_vertices = array([[0.20000, 0.50000, 0.50000],
                                        [0.50000, -0.10000, 0.50000],
                                        [0.50000, 0.50000, -0.60000],
                                        [0.50000, -0.10000, -0.60000],
                                        [-0.30000, 0.50000, 0.50000],
                                        [-0.30000, -0.10000, 0.50000],
                                        [-0.30000, 0.50000, -0.60000],
                                        [-0.30000, -0.10000, -0.60000]])
    sigmoid_slope = 500
    h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = get_polytope_hyperplane(
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

    # Available cartesian wrench/velocity polytope

    hull2 = ConvexHull(polytope_vertices, qhull_options='Qs QJ')

    polytope_faces = hull2.simplices
    return polytope_vertices, polytope_faces


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
    print('hull2.simplices', hull2.simplices)
    print('len(hull2.simplices)', len(hull2.simplices))
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
