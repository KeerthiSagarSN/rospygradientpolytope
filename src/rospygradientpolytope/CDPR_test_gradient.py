# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:54:59 2022

@author: keerthi.sagar
"""

from WrenchMatrix import get_wrench_matrix

## Library imports here
from numpy import array,savez,load,shape,transpose,argsort,sort,meshgrid,zeros
from linearalgebra import V_unit,skew_2D
from numpy.ma import MaskedArray,compress_rowcols
import os

import polytope
## Plotting library - 2D Plot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


from linearalgebra import dist_line_line_2D,isclose,check_ndarray
from robot_functions import getHessian_2
## Collision library
from shapely.geometry import Polygon,LineString
## Declare mounting parameters


from polytope_functions_2D import get_capacity_margin, get_polytope_hyperplane

from polytope_gradient_functions_2D import Gamma_hat_gradient
from gradient_functions_2D import normal_gradient

from WrenchMatrix import get_wrench_matrix


### Optimization Model
#from OptimizationModel_CDPR_backup import OptimizationModel

from OptimizationModel_CDPR import OptimizationModel

CDPR_optimizer = OptimizationModel()

# Mounting points for the 2-DOF planar cable robot 

## MOunting points

#base_points = array([[-2,-2],[-2,2],[2,2],[2,-2]])


#base_points = array([[-4,-4],[-4,-2],[-2,-2],[-2,-4]])

base_points = array([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
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
sigmoid_slope_array = array([10.0,20.0,30.0,50])
#CDPR_optimizer.sigmoid_slope = 100

CDPR_optimizer.cartesian_dof_input = array([True,True,False,False,False,False])

CDPR_optimizer.step_size = 50
CDPR_optimizer.tol_value = 5e-2
CDPR_optimizer.lower_bound = -1e-3

#q = array([-3,-3.5])
#CDPR_optimizer.analytical_solver = True


# Capacity Margin based workspace points with feasible WFW and estimated points based on estimated
# CM

ef_estimated_array = {}
CM_estimated = {}
test_number = 101

for k in range(len(sigmoid_slope_array)):

#for k in range(3,4):
    CDPR_optimizer.sigmoid_slope = sigmoid_slope_array[k]
    ef_actual,ef_est,ef_feasible,ef_infeasible,ef_total,cm_actual,cm_est,cm_total_actual,cm_total_est = CDPR_optimizer.generate_workspace()
    

    BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/CDPR_test_results/"
    file_name = 'workspace_capacity_margin_'+str(CDPR_optimizer.sigmoid_slope) + str('_') + str(test_number)

    savez(os.path.join(BASE_PATH, file_name),cm_total_actual =cm_total_actual,cm_total_est =cm_total_est,cm_actual =cm_actual,cm_est =cm_est,ef_total=ef_total,\
        ef_actual = ef_actual,ef_est  = ef_est, ef_feasible = ef_feasible, ef_infeasible  = ef_infeasible ,sigmoid_slope_inp = CDPR_optimizer.sigmoid_slope,\
                    base_points  = base_points , pos_bounds = CDPR_optimizer.pos_bounds,desired_vertices  = CDPR_optimizer.cartesian_desired_vertices, \
                        cartesian_dof_input = CDPR_optimizer.cartesian_dof_input, height_params = CDPR_optimizer.height_params , \
                            length_params = CDPR_optimizer.length_params, qdot_min  = CDPR_optimizer.qdot_min, qdot_max = CDPR_optimizer.qdot_max)


for k in range(len(sigmoid_slope_array)):
    CDPR_optimizer.sigmoid_slope = sigmoid_slope_array[k]
    BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/CDPR_test_results/"
    #test_number = 5
    file_name = 'workspace_capacity_margin_'+str(CDPR_optimizer.sigmoid_slope) + str('_') + str(test_number)
    data = load(os.path.join(BASE_PATH, file_name)+str('.npz'))
    #file_name = 'workspace_capacity_margin_'+str(CDPR_optimizer.sigmoid_slope) + str('_') + str(test_number))

    ef_total = data['ef_total']
    ef_actual = data['ef_actual']
    ef_estimated_array[k] = data['ef_est']

    ef_feasible = data['ef_feasible']
    ef_infeasible = data['ef_infeasible']
    CM_total = data['cm_total_actual']
    CM_estimated[k] = data['cm_total_est']


print('ef_actual',ef_actual)
print('ef_total',ef_total)
print('shape of ef_total',shape(ef_total))
print('shape of ef_actual',shape(ef_actual))


print('cm_total',CM_total)
print('shape of cm',shape(CM_total))
pos_CM_total = check_ndarray(CM_total)

pos_CM_total = pos_CM_total[pos_CM_total > -CDPR_optimizer.lower_bound]
pos_CM_total = sort(pos_CM_total)
print('min of CM',min(pos_CM_total))

print('pos_CM_total',pos_CM_total)


cm_est = check_ndarray(CM_estimated[0])
pos_CM_est = cm_est[cm_est>-CDPR_optimizer.lower_bound]

pos_CM_est = sort(pos_CM_est)
print('pos_CM_est',pos_CM_est)
#for i in range(len(ef_total)):
#for j in range(len(ef_total)):
cm_est = check_ndarray(CM_estimated[1])
pos_CM_est = cm_est[cm_est>-CDPR_optimizer.lower_bound]

pos_CM_est = sort(pos_CM_est)
print('pos_CM_est 1',pos_CM_est)


cm_est = check_ndarray(CM_estimated[2])
pos_CM_est = cm_est[cm_est>-CDPR_optimizer.lower_bound]

pos_CM_est = sort(pos_CM_est)
print('pos_CM_est 2',pos_CM_est)
input('test here')
print('ef_actual',ef_actual)

q_actual = ef_actual[ef_actual[:,:,0] >= 0]
q_feasible  = ef_feasible[ef_feasible[:,:,0] >= 0]
#q_feasible = ef_feasible
q_infeasible  = ef_infeasible[ef_infeasible[:,:,0] >= 0]


print('q_actual',q_actual)
#MaskedArray()
#print('q_actual is',q_actual)
print('q_feasible is',q_feasible)
print('q_infeasible is',q_infeasible)


print('shape(q_feasible)',shape(q_feasible))
#print('q_estimated is',q_estimated)
#print('shape(0q_est)',shape(q_estimated))

#global figure 7

fig_1 = plt.figure()

ax = plt.axes(projection='3d')

CM_estimated_density = CM_estimated[3]

print('shape(CM_estimated density)',shape(CM_estimated_density))
print('shape of ef_total',shape(ef_total[:,:,0]))

#X_dens,Y_dens = meshgrid(ef_total[:,:,0],ef_total[:,:,1])
w = ax.plot_surface(ef_total[:,:,0], ef_total[:,:,1], CM_estimated_density,cmap='spring')
# change the fontsize

ax.set_xlabel('x [m]',fontsize=13)
ax.set_ylabel('y [m]',fontsize=13)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#ax.set_zlabel('$\hat{\gamma}$',fontsize=20)
#ax.set_zlabel('$\hat{\gamma}$',rotation=145)
ax.set_zlabel(r"$\hat{\gamma}$" + str(' [N]'),rotation=90,fontsize=16)






plt.show()

fig_2 = plt.figure()



ef = array([0.25,0.75])


for i in range(len(base_points)):
    x_plt = array([ef[0],base_points[i,0]])
    y_plt = array([ef[1],base_points[i,1]])
    plt.plot(x_plt,y_plt,color='cyan')

plt_feasible = plt.scatter(q_feasible[:,0], q_feasible[:,1],color = 'g',s=0.5)
plt_infeasible = plt.scatter(q_infeasible[:,0], q_infeasible[:,1],color = 'r',s=0.5)


color_array = ['darkorange','magenta','blue','white','magenta','blue']

plt_estimate = {}
plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],facecolors='none', edgecolors='k',marker = 'D',alpha=0.75,s=2.150)
for k in range(len(sigmoid_slope_array)):

    if (k<5.0):

        ef_estimated = ef_estimated_array[k]
        q_estimated = ef_estimated[ef_estimated[:,:,0] >= 0]
        print('q_estimated',q_estimated)

        #q_actual_sort = argsort(q_actual,axis=0)
        #q_estimated_sort = argsort(q_estimated,axis=0)
        #plt_actual = plt.plot(q_actual[:,0], q_actual[:,1],color = 'k')
        
        #plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],color = 'k',s=2.0)
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        plt_estimate[k] = plt.scatter(q_estimated[:,0], q_estimated[:,1],edgecolors='none',facecolors = color_array[k],marker = 'o',alpha=0.75,s=2.5)
        
    else:

        q_estimated = ef_estimated_array[k]  

        #q_actual_sort = argsort(q_actual,axis=0)
        #q_estimated_sort = argsort(q_estimated,axis=0)
        #plt_actual = plt.plot(q_actual[:,0], q_actual[:,1],color = 'k')
        
        #plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],color = 'k',s=2.0)
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        plt_estimate[k] = plt.scatter(q_estimated[:,0], q_estimated[:,1],facecolor = color_array[k],s=2.1)


        #plt_estimate = plt_actual







plt.scatter(base_points [:,0], base_points [:,1],marker='s',facecolors='none', edgecolors='k')
plt.scatter(ef[0], ef[1],marker='o',color='k')

#plt.savefig()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

x0 = array([0.24,0.42])
step_size = 0.0001
number_iterations = 10000


#@staticmethod
#def test_normal_gradient_2D(self,x0,step_size,number_iterations):

from scipy.linalg import qr
from numpy.linalg import norm
print('test_normal_gradient')
CDPR_optimizer.active_joints = 4
test_cable = 2
q = x0
first_iteration = True
for i in range(number_iterations):
    #q[0] += step_size
    q[1] += step_size
    x_in = q[0]
    y_in = q[1]
    Wm = zeros(shape=(2,CDPR_optimizer.active_joints))
    n_k = zeros(shape = (shape(Wm)[1],2))
    for k in range(len(CDPR_optimizer.base_points)):
        cable_plt = array([[x_in,CDPR_optimizer.base_points[k,0]],[y_in,CDPR_optimizer.base_points[k,1]]])
        Wm[0,k] = CDPR_optimizer.base_points[k,0] - x_in
        Wm[1,k] = CDPR_optimizer.base_points[k,1] - y_in

        #print('self.base_points',self.base_points[k,:])

        #input('stop and check')

        #Wm[0,k] = W[0,k]*((norm(W[:,k]))**(-1))
        #Wm[1,k] = W[1,k]*((norm(W[:,k]))**(-1))

        Wm[:,k] = V_unit(Wm[:,k])
        #plt.plot(cable_plt[0,:],cable_plt[1,:],color = color_arr[k])
        #plt.pause(0.01)
        #print('cable number is:',k)
    #print('Wrench matrix is is',W)
    
    #W = W
    
    #print('Wrench matrix is here', W)
    #input('wait here')

    
    #input('stop here')
    #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
    #Wm = array([[-0.7071,-0.7071,-0.7071,-0.7071],[0.7071,0.7071,0.7071,0.7071]])
    
    #print('Wrench matrix is is',Wm)

    W = -Wm

    print('W outside by default',W)
    #Wu,Wn,Jx,Jy= get_wrench_matrix(q,CDPR_optimizer.length_params,CDPR_optimizer.height_params)

    Wu,Wn,Hess= get_wrench_matrix(q,CDPR_optimizer.length_params,CDPR_optimizer.height_params)
    #print('jacobian is',J(x_in,y_in))
    
    


    for l in range(shape(W)[1]):
        Wmk = array([W[:,l]])
        print('Wmk is',Wmk)
        Q_mat,R_mat = qr(transpose(Wmk))

        #print('Q_mat',Q_mat)
        #n_k[l,0] = Q_mat[0,-1]
        #n_k[l,1] = Q_mat[1,-1]
        #print('n_k with qr decomp is',n_k[l,:])
        n_k[l,:] = skew_2D(Wmk)
        #print('n_k with skew matrix is',n_k[l,:])
        #input('check skew matrix')
    print('n_k outside the loop is',n_k)
    
    #dnk_dq = normal_qr_gradient(-Wn,-H)
    #du_dq_a = H[:,:,1]
    #dnk_dq_a = dnk_dq[:,:,1]
    #dnk_dq_a = H3(x_in,y_in)
    #dnk_dq_a = JY(x_in,y_in)
    #dnk_dq = getHessian_2(W)
    #print('dnk_dq',dnk_dq)
    dnk_dq_a = normal_gradient(Hess)
    #print('dnk_dq_a')
    #dnk_dq_a = Jy(x_in,y_in)
    print('Hess',Hess)
    dnk_dq_a = dnk_dq_a[:,:,1]
    print('Jx',dnk_dq_a)

    if not first_iteration:
        dn_dq_n = (n_k - prev_n_k)/(step_size*1.0)

        
        
        print('dn_dq')
        print('Analytical_gradient',dnk_dq_a)
        print('NUmerical gradient',dn_dq_n)
        print('Error')
        print(norm(dn_dq_n - transpose(dnk_dq_a)))
        input('check gradient matrix')



        prev_n_k = n_k
        
    if first_iteration:
        prev_n_k = n_k
        prev_twist = Wn
        first_iteration = False





print('test_normal_gradient')

'''
def gradient_descent(ax, x0, learning_rate, num_iterations):
    q = x0
    for i in range(num_iterations):
        grad = df(q)
        q -= learning_rate * grad
        ax.scatter3D(q[0],q[1],c='white',s=10.0)

    return q

# Define the function f(x) and its gradient df(x)
def func_opt(q):

    x_in = q[0]
    y_in = q[1]
    Wm = zeros(shape=(2,CDPR_optimizer.active_joints))
    for k in range(len(CDPR_optimizer.base_points)):
        cable_plt = array([[x_in,CDPR_optimizer.base_points[k,0]],[y_in,CDPR_optimizer.base_points[k,1]]])
        Wm[0,k] = CDPR_optimizer.base_points[k,0] - x_in
        Wm[1,k] = CDPR_optimizer.base_points[k,1] - y_in

        #print('self.base_points',self.base_points[k,:])

        #input('stop and check')

        #Wm[0,k] = W[0,k]*((norm(W[:,k]))**(-1))
        #Wm[1,k] = W[1,k]*((norm(W[:,k]))**(-1))

        Wm[:,k] = V_unit(Wm[:,k])
        #plt.plot(cable_plt[0,:],cable_plt[1,:],color = color_arr[k])
        #plt.pause(0.01)
        #print('cable number is:',k)
    #print('Wrench matrix is is',W)
    
    #W = W
    
    #print('Wrench matrix is here', W)
    #input('wait here')

    
    #input('stop here')
    #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
    #Wm = array([[-0.7071,-0.7071,-0.7071,-0.7071],[0.7071,0.7071,0.7071,0.7071]])
    
    #print('Wrench matrix is is',Wm)
    W = -Wm



    h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
    get_polytope_hyperplane(W,CDPR_optimizer.active_joints,CDPR_optimizer.cartesian_dof_input,CDPR_optimizer.qdot_min,\
        CDPR_optimizer.qdot_max,CDPR_optimizer.sigmoid_slope)
                    
                
    Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = \
    get_capacity_margin(W,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
            CDPR_optimizer.active_joints,CDPR_optimizer.cartesian_dof_input,CDPR_optimizer.qdot_min,CDPR_optimizer.qdot_max,CDPR_optimizer.cartesian_desired_vertices,CDPR_optimizer.sigmoid_slope)

    return Gamma_min_softmax
def df(q):

    Wm = zeros(shape=(2,self.active_joints))
    for k in range(len(self.base_points)):
        cable_plt = array([[x_in,self.base_points[k,0]],[y_in,self.base_points[k,1]]])
        Wm[0,k] = self.base_points[k,0] - x_in
        Wm[1,k] = self.base_points[k,1] - y_in

        #print('self.base_points',self.base_points[k,:])

        #input('stop and check')

        #Wm[0,k] = W[0,k]*((norm(W[:,k]))**(-1))
        #Wm[1,k] = W[1,k]*((norm(W[:,k]))**(-1))

        Wm[:,k] = V_unit(Wm[:,k])
        #plt.plot(cable_plt[0,:],cable_plt[1,:],color = color_arr[k])
        #plt.pause(0.01)
        #print('cable number is:',k)
    #print('Wrench matrix is is',W)
    
    #W = W
    
    #print('Wrench matrix is here', W)
    #input('wait here')

    
    #input('stop here')
    #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
    #Wm = array([[-0.7071,-0.7071,-0.7071,-0.7071],[0.7071,0.7071,0.7071,0.7071]])
    
    #print('Wrench matrix is is',Wm)
    W = -Wm



    h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
    get_polytope_hyperplane(W,CDPR_optimizer.active_joints,CDPR_optimizer.cartesian_dof_input,CDPR_optimizer.qdot_min,\
        CDPR_optimizer.qdot_max,CDPR_optimizer.sigmoid_slope)
                    
                
    Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = \
    get_capacity_margin(W,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
            CDPR_optimizer.active_joints,CDPR_optimizer.cartesian_dof_input,CDPR_optimizer.qdot_min,CDPR_optimizer.qdot_max,CDPR_optimizer.cartesian_desired_vertices,CDPR_optimizer.sigmoid_slope)



    d_gamma_hat = Gamma_hat_gradient(JE,H,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                        p_minus_hat,Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat,\
                        CDPR_optimizer.qdot_min,CDPR_optimizer.qdot_max,CDPR_optimizer.cartesian_desired_vertices,CDPR_optimizer.sigmoid_slope)
    return d_gamma_hat


'''

## Secondary plot
'''
fig_2 = plt.figure()

ef = array([0.25,0.75])


for i in range(len(base_points)):
    x_plt = array([ef[0],base_points[i,0]])
    y_plt = array([ef[1],base_points[i,1]])
    plt.plot(x_plt,y_plt,color='cyan')

plt_feasible = plt.scatter(q_feasible[:,0], q_feasible[:,1],color = 'g',s=0.0005)
plt_infeasible = plt.scatter(q_infeasible[:,0], q_infeasible[:,1],color = 'r',s=0.0005)


color_array = ['darkorange','magenta','blue','white','magenta','blue']

plt_estimate = {}
plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],facecolors='none', edgecolors='k',marker = 'D',alpha=0.75,s=0.150)
for k in range(len(sigmoid_slope_array)):

    if (k<5.0):

        ef_estimated = ef_estimated_array[k]
        q_estimated = ef_estimated[ef_estimated[:,:,0] >= 0]
        print('q_estimated',q_estimated)

        #q_actual_sort = argsort(q_actual,axis=0)
        #q_estimated_sort = argsort(q_estimated,axis=0)
        #plt_actual = plt.plot(q_actual[:,0], q_actual[:,1],color = 'k')
        
        #plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],color = 'k',s=2.0)
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        plt_estimate[k] = plt.scatter(q_estimated[:,0], q_estimated[:,1],edgecolors='none',facecolors = color_array[k],marker = 'o',alpha=0.75,s=0.5)
        
    else:

        q_estimated = ef_estimated_array[k]  

        #q_actual_sort = argsort(q_actual,axis=0)
        #q_estimated_sort = argsort(q_estimated,axis=0)
        #plt_actual = plt.plot(q_actual[:,0], q_actual[:,1],color = 'k')
        
        #plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],color = 'k',s=2.0)
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        plt_estimate[k] = plt.scatter(q_estimated[:,0], q_estimated[:,1],facecolor = color_array[k],s=0.1)


        #plt_estimate = plt_actual







plt.scatter(base_points [:,0], base_points [:,1],marker='s',facecolors='none', edgecolors='k')
plt.scatter(ef[0], ef[1],marker='o',color='k')

#plt.savefig()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig('Wrench Feasible Workspace_' + str(CDPR_optimizer.sigmoid_slope)+('.png'),dpi=600)




















plt_empty = plt.scatter(ef[0],ef[1],color='g',s=0.0000001)

plt_feasible = plt.scatter(q_feasible[:,0], q_feasible[:,1],color = 'g',s=0.1001)
plt_infeasible = plt.scatter(q_infeasible[:,0], q_infeasible[:,1],color = 'r',s=0.1001)




for k in range(len(sigmoid_slope_array)):

    if (k<5.0):

        ef_estimated = ef_estimated_array[k]
        q_estimated = ef_estimated[ef_estimated[:,:,0] >= 0]
        print('q_estimated',q_estimated)

        #q_actual_sort = argsort(q_actual,axis=0)
        #q_estimated_sort = argsort(q_estimated,axis=0)
        #plt_actual = plt.plot(q_actual[:,0], q_actual[:,1],color = 'k')
        
        #plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],color = 'k',s=2.0)
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        plt_estimate[k] = plt.scatter(q_estimated[:,0], q_estimated[:,1],facecolor = color_array[k],marker='o',s=5.0)
        
    else:

        q_estimated = ef_estimated_array[k]  

        #q_actual_sort = argsort(q_actual,axis=0)
        #q_estimated_sort = argsort(q_estimated,axis=0)
        #plt_actual = plt.plot(q_actual[:,0], q_actual[:,1],color = 'k')
        
        #plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],color = 'k',s=2.0)
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        plt_estimate[k] = plt.scatter(q_estimated[:,0], q_estimated[:,1],facecolor = color_array[k],s=1.001)


plt.xlim(0,0.26)
plt.ylim(0.70,0.90)
plt.scatter(ef[0], ef[1],marker='o',color='k')
plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],facecolors='none', edgecolors='k',marker = 'D',alpha=0.5,s=10)
plt.legend((plt_feasible,plt_infeasible,plt_actual,plt_empty ,plt_estimate[0],plt_estimate[1],plt_estimate[2],plt_estimate[3]),\
    ('WFW','Wrench Infeasbible', 'Actual Boundary','Estimated Boundary','Slope-' + str(sigmoid_slope_array[0]),\
        'Slope-' + str(sigmoid_slope_array[1]),'Slope-' + str(sigmoid_slope_array[2]),\
           'Slope-' + str(sigmoid_slope_array[3]))\
                ,markerscale=4,loc=3)
plt.savefig('Wrench Feasible Workspace Zoomed_' + str(CDPR_optimizer.sigmoid_slope)+('.png'),dpi=600)
plt.show()
'''