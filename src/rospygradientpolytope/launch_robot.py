# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:23:34 2022

@author: keerthi.sagar

Launch a Serial/Tree robot with URDF

1. Instantiate Robot parameter Server here
2. Publish and Subscribe to the robot joints
3. Call IK solver based on Scipy.optimize - Constraint Function/ Objective Function (Maximize Capacity Margin)



File is inspired by the Architecture of Robot Solver of Antun Skuric

https://github.com/askuric/polytope_vertex_search/blob/master/ROS_nodes/panda_capacity/scripts/capacity/robot_solver.py
"""

