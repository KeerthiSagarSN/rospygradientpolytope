# -*- coding: utf-8 -*-
"""
Created on Thur Sep 08 12:52:25 2022

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


# Mounting points for the 2-DOF planar cable robot 

## MOunting points

base_points = array([[-2,-1],[-2,1],[2,1],[2,-1]])



### Obstacle definitions

obstacle_polytope_1 = array([[0.3,0.5],[0.3,0.0],[-0.05,0.0],[-0.05,0.5]])




obstacle_polytope_2 = array([[-0.5,-0.1],[-0.5,0.0],[0.25,0.0],[0.25,-0.1]])
## Platform points - point mass


# Plot the obstacle with cable and base_points
plt.figure()



for x_pos in range(-5,5) :
    for y_pos in range(-5,5):
        
        ##### Obtsacle plot
        x = [obstacle_polytope_1[0,0],obstacle_polytope_1[1,0],obstacle_polytope_1[2,0],obstacle_polytope_1[3,0],obstacle_polytope_1[0,0]]
        y = [obstacle_polytope_1[0,1],obstacle_polytope_1[1,1],obstacle_polytope_1[2,1],obstacle_polytope_1[3,1],obstacle_polytope_1[0,1]]
        plt.plot(x,y,color = 'k')
        
        x = [obstacle_polytope_2[0,0],obstacle_polytope_2[1,0],obstacle_polytope_2[2,0],obstacle_polytope_2[3,0],obstacle_polytope_2[0,0]]
        y = [obstacle_polytope_2[0,1],obstacle_polytope_2[1,1],obstacle_polytope_2[2,1],obstacle_polytope_2[3,1],obstacle_polytope_2[0,1]]
        plt.plot(x,y,color = 'k')
        x = x_pos*0.1
        y = y_pos*0.15

        q = array([x,y])
        print('q',q)
        
        c1 = LineString([q,base_points[0]])
        c2 = LineString([q,base_points[1]])
        c3 = LineString([q,base_points[2]])
        c4 = LineString([q,base_points[3]])
        
        
        obs1 = Polygon(obstacle_polytope_1)
        obs2 = Polygon(obstacle_polytope_2)
        print(obs1.intersects(c1))
        
        cable_color = 'b'
        '''
        if obs1.intersects(c1) or obs2.intersects(c1):
            cable_color = 'r'
        '''
        
        if isclose(dist_line_line_2D(q , base_points[0],obstacle_polytope_1[0],obstacle_polytope_1[1]),0.0,1e-3) or \
            isclose(dist_line_line_2D(q , base_points[0],obstacle_polytope_1[1],obstacle_polytope_1[2]),0.0,1e-3) or \
                isclose(dist_line_line_2D(q , base_points[0],obstacle_polytope_1[2],obstacle_polytope_1[3]),0.0,1e-3) or \
                    isclose(dist_line_line_2D(q , base_points[0], obstacle_polytope_1[3],obstacle_polytope_1[0]),0.0,1e-3) or \
                        isclose(dist_line_line_2D(q , base_points[0],obstacle_polytope_2[0],obstacle_polytope_2[1]),0.0,1e-3) or \
                            isclose(dist_line_line_2D(q , base_points[0],obstacle_polytope_2[1],obstacle_polytope_2[2]),0.0,1e-3) or \
                                isclose(dist_line_line_2D(q , base_points[0],obstacle_polytope_2[2],obstacle_polytope_2[3]),0.0,1e-3) or \
                                    isclose(dist_line_line_2D(q , base_points[0],obstacle_polytope_2[3],obstacle_polytope_2[0]),0.0,1e-3):
                                        cable_color = 'r'
        
        
        x = [q[0],base_points[0,0]]
        y = [q[1],base_points[0,1]]
        plt.plot(x,y,color = cable_color)
        
        cable_color = 'b'
        '''
        if obs1.intersects(c2) or obs2.intersects(c2):
            cable_color = 'r'
        '''
        if isclose(dist_line_line_2D(q , base_points[1],obstacle_polytope_1[0],obstacle_polytope_1[1]),0.0,1e-3) or \
            isclose(dist_line_line_2D(q , base_points[1],obstacle_polytope_1[1],obstacle_polytope_1[2]),0.0,1e-3) or \
                isclose(dist_line_line_2D(q , base_points[1],obstacle_polytope_1[2],obstacle_polytope_1[3]),0.0,1e-3) or \
                    isclose(dist_line_line_2D(q , base_points[1], obstacle_polytope_1[3],obstacle_polytope_1[0]),0.0,1e-3) or \
                        isclose(dist_line_line_2D(q , base_points[1],obstacle_polytope_2[0],obstacle_polytope_2[1]),0.0,1e-3) or \
                            isclose(dist_line_line_2D(q , base_points[1],obstacle_polytope_2[1],obstacle_polytope_2[2]),0.0,1e-3) or \
                                isclose(dist_line_line_2D(q , base_points[1],obstacle_polytope_2[2],obstacle_polytope_2[3]),0.0,1e-3) or \
                                    isclose(dist_line_line_2D(q , base_points[1],obstacle_polytope_2[3],obstacle_polytope_2[0]),0.0,1e-3):
                                        cable_color = 'r'
        

        
        x = [q[0],base_points[1,0]]
        y = [q[1],base_points[1,1]]
        plt.plot(x,y,color = cable_color)
        
        cable_color = 'b'
        '''
        if obs1.intersects(c3) or obs2.intersects(c3):
            cable_color = 'r'
        '''
        if isclose(dist_line_line_2D(q , base_points[2],obstacle_polytope_1[0],obstacle_polytope_1[1]),0.0,1e-3) or \
            isclose(dist_line_line_2D(q , base_points[2],obstacle_polytope_1[1],obstacle_polytope_1[2]),0.0,1e-3) or \
                isclose(dist_line_line_2D(q , base_points[2],obstacle_polytope_1[2],obstacle_polytope_1[3]),0.0,1e-3) or \
                    isclose(dist_line_line_2D(q , base_points[2], obstacle_polytope_1[3],obstacle_polytope_1[0]),0.0,1e-3) or \
                        isclose(dist_line_line_2D(q , base_points[2],obstacle_polytope_2[0],obstacle_polytope_2[1]),0.0,1e-3) or \
                            isclose(dist_line_line_2D(q , base_points[2],obstacle_polytope_2[1],obstacle_polytope_2[2]),0.0,1e-3) or \
                                isclose(dist_line_line_2D(q , base_points[2],obstacle_polytope_2[2],obstacle_polytope_2[3]),0.0,1e-3) or \
                                    isclose(dist_line_line_2D(q , base_points[2],obstacle_polytope_2[3],obstacle_polytope_2[0]),0.0,1e-3):
                                        cable_color = 'r'
        
        

        x = [q[0],base_points[2,0]]
        y = [q[1],base_points[2,1]]
        plt.plot(x,y,color = cable_color)
        
        cable_color = 'b'
        '''
        if obs1.intersects(c4) or obs2.intersects(c4):
            cable_color = 'r'
        '''
        if isclose(dist_line_line_2D(q , base_points[3],obstacle_polytope_1[0],obstacle_polytope_1[1]),0.0,1e-3) or \
            isclose(dist_line_line_2D(q , base_points[3],obstacle_polytope_1[1],obstacle_polytope_1[2]),0.0,1e-3) or \
                isclose(dist_line_line_2D(q , base_points[3],obstacle_polytope_1[2],obstacle_polytope_1[3]),0.0,1e-3) or \
                    isclose(dist_line_line_2D(q , base_points[3], obstacle_polytope_1[3],obstacle_polytope_1[0]),0.0,1e-3) or \
                        isclose(dist_line_line_2D(q , base_points[3],obstacle_polytope_2[0],obstacle_polytope_2[1]),0.0,1e-3) or \
                            isclose(dist_line_line_2D(q , base_points[3],obstacle_polytope_2[1],obstacle_polytope_2[2]),0.0,1e-3) or \
                                isclose(dist_line_line_2D(q , base_points[3],obstacle_polytope_2[2],obstacle_polytope_2[3]),0.0,1e-3) or \
                                    isclose(dist_line_line_2D(q , base_points[3],obstacle_polytope_2[3],obstacle_polytope_2[0]),0.0,1e-3):
                                        cable_color = 'r'
        

        x = [q[0],base_points[3,0]]
        y = [q[1],base_points[3,1]]
        plt.plot(x,y,color = cable_color)
        
        plt.pause(0.5)
        plt.cla()
        #plot()
        #W,Hessian_matrix = get_wrench_matrix(q,4.0,2.0)

        
