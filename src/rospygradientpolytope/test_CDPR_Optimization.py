# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:54:59 2022

@author: keerthi.sagar
"""

from WrenchMatrix import get_wrench_matrix

## Library imports here
from numpy import array

## Plotting library - 2D Plot

import matplotlib.pyplot as plt

import random

from linearalgebra import dist_line_line_2D,isclose
## Collision library
from shapely.geometry import Polygon,LineString
## Declare mounting parameters


### Optimization Model
from OptimizationModel_CDPR import OptimizationModel

CDPR_optimizer = OptimizationModel()

# Mounting points for the 2-DOF planar cable robot 

## MOunting points

base_points = array([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
CDPR_optimizer.base_points = base_points



### Obstacle definitions
CDPR_optimizer.base_points = base_points



### Obstacle definitions

'''

obstacle_polytope_1 = array([[0.3,0.5],[0.3,0.0],[-0.05,0.0],[-0.05,0.5]])
obstacle_polytope_2 = array([[-0.95,-0.6],[-0.95,-0.2],[-0.05,-0.2],[-0.05,-0.6]])

CDPR_optimizer.obstacle_set = array([obstacle_polytope_1,obstacle_polytope_2])
'''
#CDPR_optimizer.roi_center = array([0,0])

CDPR_optimizer.pos_bounds = array([[min(base_points[:,0])+0.005,max(base_points[:,0])-0.005],[min(base_points[:,1])+0.005,max(base_points[:,1])-0.005]])
print('CDPR_optimizer.pos_bounds',CDPR_optimizer.pos_bounds)
#CDPR_optimizer.cartesian_desired_vertices = 1*array([[1,10],[3,-3],[60,1],[30,10],[10,10]])

input('test here')


#CDPR_optimizer.cartesian_desired_vertices = 1*array([[3,3],[1,10]])
#CDPR_optimizer.cartesian_desired_vertices = array([[0,-9.8*5]])

CDPR_optimizer.cartesian_desired_vertices = 2.0*array([[-5.0,-5.0],[-5.0,5.0],[5.0,5.0],[5.0,-5.0]])


#CDPR_optimizer.cartesian_desired_vertices = 1.0*array([[0.0,0.0],[0.0,-50.0]])
CDPR_optimizer.qdot_max = 1.0*array([25.0,25.0,25.0,25.0])
CDPR_optimizer.qdot_min = 1.0*array([1.0,1.0,1.0,1.0])

CDPR_optimizer.length_params = base_points[2,0] - base_points[0,0]
CDPR_optimizer.height_params = base_points[2,1] - base_points[0,1]

print('Length is ', CDPR_optimizer.length_params )
print('Height is ', CDPR_optimizer.height_params )
#sigmoid_slope_array = array([10.0,50.0,100.0,150.0,400.0])
#sigmoid_slope_array = array([10.0,20.0,30.0,50]) ### Paper value here
sigmoid_slope_array = array([50.0,50.0,50.0,50.0]) ### Paper value here
#CDPR_optimizer.sigmoid_slope = 100

CDPR_optimizer.cartesian_dof_input = array([True,True,False,False,False,False])
CDPR_optimizer.sigmoid_slope = sigmoid_slope_array[0]
CDPR_optimizer.step_size = 50
CDPR_optimizer.tol_value = 5e-2
CDPR_optimizer.lower_bound = -1e-3


obstacle_polytope_1 = array([[0.3,0.3],[0.3,0.5],[0.42,0.5],[0.42,0.3]])
obstacle_polytope_2 = array([[0.71,0.71],[0.71,0.81],[0.81,0.81],[0.81,0.71]])

CDPR_optimizer.obstacle_set = array([obstacle_polytope_1,obstacle_polytope_2])

CDPR_optimizer.roi_center = array([0.5,0.5])

#CDPR_optimizer.pos_bounds = array([[0,1],[0,1]])

optimizer_success = False
while not optimizer_success:


    q = array([random.uniform(CDPR_optimizer.pos_bounds[0,0],CDPR_optimizer.pos_bounds[0,1]),random.uniform(CDPR_optimizer.pos_bounds[1,0],CDPR_optimizer.pos_bounds[1,1])])
    CDPR_optimizer.analytical_solver = True



    CDPR_optimizer.fmin_opt(q)

    x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
    y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
    plt.plot(x,y,color = 'k')

    x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
    y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
    plt.plot(x,y,color = 'k')

    ef = CDPR_optimizer.q_joints_opt.x
    optimizer_success  = CDPR_optimizer.q_joints_opt.success
    print('Status',CDPR_optimizer.q_joints_opt)

    cable_color = 'g'
    for c in range(len(base_points)):
        x = [ef[0],base_points[c,0]]
        y = [ef[1],base_points[c,1]]
        plt.plot(x,y,color = cable_color)