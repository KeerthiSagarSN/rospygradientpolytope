# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:54:59 2022

@author: keerthi.sagar
"""

from WrenchMatrix import get_wrench_matrix

## Library imports here
from numpy import array,savez,load,shape,transpose
import os

import polytope
## Plotting library - 2D Plot

import matplotlib.pyplot as plt

from linearalgebra import dist_line_line_2D,isclose
## Collision library
from shapely.geometry import Polygon,LineString
## Declare mounting parameters


### Optimization Model
#from OptimizationModel_CDPR_backup import OptimizationModel

from OptimizationModel_CDPR import OptimizationModel

CDPR_optimizer = OptimizationModel()

# Mounting points for the 2-DOF planar cable robot 

## MOunting points

#base_points = array([[-2,-2],[-2,2],[2,2],[2,-2]])


#base_points = array([[-4,-4],[-4,-2],[-2,-2],[-2,-4]])

base_points = array([[0,0],[0,1],[1,1],[1,0]])
CDPR_optimizer.base_points = base_points



### Obstacle definitions

'''

obstacle_polytope_1 = array([[0.3,0.5],[0.3,0.0],[-0.05,0.0],[-0.05,0.5]])
obstacle_polytope_2 = array([[-0.95,-0.6],[-0.95,-0.2],[-0.05,-0.2],[-0.05,-0.6]])

CDPR_optimizer.obstacle_set = array([obstacle_polytope_1,obstacle_polytope_2])
'''
#CDPR_optimizer.roi_center = array([0,0])

CDPR_optimizer.pos_bounds = array([[min(base_points[:,0]),max(base_points[:,0])],[min(base_points[:,1]),max(base_points[:,1])]])
print('CDPR_optimizer.pos_bounds',CDPR_optimizer.pos_bounds)
#CDPR_optimizer.cartesian_desired_vertices = 1*array([[1,10],[3,-3],[60,1],[30,10],[10,10]])

input('test here')


#CDPR_optimizer.cartesian_desired_vertices = 1*array([[3,3],[1,10]])
CDPR_optimizer.cartesian_desired_vertices = array([[0,-9.8*5]])

CDPR_optimizer.qdot_max = 1*array([25,25,25,25])
CDPR_optimizer.qdot_min = 0*array([0,0,0,0])

CDPR_optimizer.length_params = base_points[2,0] - base_points[0,0]
CDPR_optimizer.height_params = base_points[2,1] - base_points[0,1]

CDPR_optimizer.sigmoid_slope = 200

CDPR_optimizer.cartesian_dof_input = array([True,True,False,False,False,False])

CDPR_optimizer.step_size = 100

#q = array([-3,-3.5])
#CDPR_optimizer.analytical_solver = True


# Capacity Margin based workspace points with feasible WFW and estimated points based on estimated
# CM

ef_actual,ef_est,ef_feasible,ef_infeasible = CDPR_optimizer.generate_workspace()
test_number = 6

BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/CDPR_test_results/"
file_name = 'workspace_capacity_margin_'+str(CDPR_optimizer.sigmoid_slope) + str('_') + str(test_number)

savez(os.path.join(BASE_PATH, file_name),ef_actual = ef_actual,ef_est  = ef_est, ef_feasible = ef_feasible, ef_infeasible  = ef_infeasible ,sigmoid_slope_inp = CDPR_optimizer.sigmoid_slope,\
                base_points  = base_points , pos_bounds = CDPR_optimizer.pos_bounds,desired_vertices  = CDPR_optimizer.cartesian_desired_vertices, \
                    cartesian_dof_input = CDPR_optimizer.cartesian_dof_input, height_params = CDPR_optimizer.height_params , \
                        length_params = CDPR_optimizer.length_params, qdot_min  = CDPR_optimizer.qdot_min, qdot_max = CDPR_optimizer.qdot_max)


BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/CDPR_test_results/"
#test_number = 5
file_name = 'workspace_capacity_margin_'+str(CDPR_optimizer.sigmoid_slope) + str('_') + str(test_number)
data = load(os.path.join(BASE_PATH, file_name)+str('.npz'))
#file_name = 'workspace_capacity_margin_'+str(CDPR_optimizer.sigmoid_slope) + str('_') + str(test_number))


ef_actual = data['ef_actual']
ef_estimated = data['ef_est']

ef_feasible = data['ef_feasible']
ef_infeasible = data['ef_infeasible']


q_actual = ef_actual

q_estimated = ef_estimated

q_feasible = ef_feasible
q_infeasible = ef_infeasible

#print('q_actual is',q_actual)
#print('shape(q_actual)',shape(q_actual))
#print('q_estimated is',q_estimated)
#print('shape(q_est)',shape(q_estimated))

#global figure 7

plt.plot(base_points [:,0], base_points [:,1],color = 'cyan')

plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],color = 'k',s=2.0)
plt_estimate = plt.scatter(q_estimated[:,0], q_estimated[:,1],color = 'cyan',s=2.5)

#plt_estimate = plt_actual

plt_feasible = plt.scatter(q_feasible[:,0], q_feasible[:,1],color = 'green',s=1.11)
plt_infeasible = plt.scatter(q_infeasible[:,0], q_infeasible[:,1],color = 'red',s=1.11)






plt.legend((plt_feasible,plt_infeasible,plt_actual,plt_estimate),('WFW','Wrench Infeasbible', 'Capacity Margin Boundary - Actual','Capacity Margin Boundary - Estimated'),markerscale=4)
#plt.savefig()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig('Wrench Feasible Workspace_' + str(CDPR_optimizer.sigmoid_slope)+('.png'))
plt.show()
'''
global figure3
figure3 = plt.figure()
plt.plot(CDPR_optimizer.cartesian_desired_vertices [:,0], CDPR_optimizer.cartesian_desired_vertices [:,1],color = 'red')
desired_vertex_set = plt.scatter(CDPR_optimizer.cartesian_desired_vertices [:,0], CDPR_optimizer.cartesian_desired_vertices [:,1],color = 'k')


plt.xlabel('Fx (N)')
plt.ylabel('Fy (N)')
plt.savefig('Cartesian_poltyope_' + str(CDPR_optimizer.sigmoid_slope)+('.png'))
plt.show()
'''





'''

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
'''