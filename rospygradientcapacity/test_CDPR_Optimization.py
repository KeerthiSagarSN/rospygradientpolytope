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

from linearalgebra import dist_line_line_2D,isclose
## Collision library
from shapely.geometry import Polygon,LineString
## Declare mounting parameters


### Optimization Model
from OptimizationModel_CDPR import OptimizationModel

CDPR_optimizer = OptimizationModel()

# Mounting points for the 2-DOF planar cable robot 

## MOunting points

base_points = array([[-2,-1],[-2,1],[2,1],[2,-1]])
CDPR_optimizer.base_points = base_points



### Obstacle definitions



obstacle_polytope_1 = array([[0.3,0.5],[0.3,0.0],[-0.05,0.0],[-0.05,0.5]])
obstacle_polytope_2 = array([[-0.95,-0.6],[-0.95,-0.2],[-0.05,-0.2],[-0.05,-0.6]])

CDPR_optimizer.obstacle_set = array([obstacle_polytope_1,obstacle_polytope_2])

CDPR_optimizer.roi_center = array([0,0])

CDPR_optimizer.pos_bounds = array([[-3,3],[-3,3]])

q = array([-3,-3.5])
CDPR_optimizer.analytical_solver = False



CDPR_optimizer.fmin_opt(q)

x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
plt.plot(x,y,color = 'k')

x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
plt.plot(x,y,color = 'k')

ef = CDPR_optimizer.q_joints_opt.x

cable_color = 'g'
for c in range(len(base_points)):
    x = [ef[0],base_points[c,0]]
    y = [ef[1],base_points[c,1]]
    plt.plot(x,y,color = cable_color)