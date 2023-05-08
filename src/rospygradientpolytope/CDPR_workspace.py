# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:54:59 2022

@author: keerthi.sagar
"""

from WrenchMatrix import get_wrench_matrix

## Library imports here
from numpy import array,savez,load,shape,transpose,argsort,sort,meshgrid
from numpy.ma import MaskedArray,compress_rowcols
import os

import polytope
## Plotting library - 2D Plot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


from linearalgebra import dist_line_line_2D,isclose,check_ndarray
## Collision library
from shapely.geometry import Polygon,LineString
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import nearest_points
## Declare mounting parameters


import scipy.optimize as sco

from numpy import array,pi,vstack,linspace,shape,zeros,hstack,transpose,matmul
from numpy.linalg import norm
from numpy.random import randn

from shapely.geometry import Polygon,LineString
from linearalgebra import isclose,V_unit
import matplotlib.pyplot as plt
from polytope_functions_2D import get_capacity_margin, get_polytope_hyperplane
from polytope_gradient_functions_2D import Gamma_hat_gradient_2D,hyperplane_gradient_2D
from gradient_functions_2D import normal_gradient,sigmoid_gradient
from WrenchMatrix import get_wrench_matrix
from robot_functions import sigmoid


from scipy.spatial.distance import directed_hausdorff

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

#input('test here')


#CDPR_optimizer.cartesian_desired_vertices = 1*array([[3,3],[1,10]])
#CDPR_optimizer.cartesian_desired_vertices = array([[0,-9.8*5]])

CDPR_optimizer.cartesian_desired_vertices = 2.0*array([[-5.0,-5.0],[-5.0,5.0],[5.0,5.0],[5.0,-5.0]])


#CDPR_optimizer.cartesian_desired_vertices = 1.0*array([[0.0,0.0],[0.0,-50.0]])
CDPR_optimizer.qdot_max = 1.0*array([25.0,25.0,25.0,25.0])
CDPR_optimizer.qdot_min = 1.0*array([1.0,1.0,1.0,1.0])

CDPR_optimizer.length_params = base_points[2,0] - base_points[0,0]
CDPR_optimizer.height_params = base_points[2,1] - base_points[0,1]


#sigmoid_slope_array = array([10.0,50.0,100.0,150.0,400.0])
sigmoid_slope_array = array([2.0,5.0,10.0,20.0,30.0,50.0])
#sigmoid_slope_array = array([10.0,20.0,30.0,50.0])
#CDPR_optimizer.sigmoid_slope = 100

CDPR_optimizer.cartesian_dof_input = array([True,True,False,False,False,False])

CDPR_optimizer.step_size = 1000
CDPR_optimizer.tol_value = 5e-2
CDPR_optimizer.lower_bound = 1e-3 #Paper value

CDPR_optimizer.active_joints = 4

#q = array([-3,-3.5])
#CDPR_optimizer.analytical_solver = True


# Capacity Margin based workspace points with feasible WFW and estimated points based on estimated
# CM

ef_estimated_array = {}
ef_actual = {}
CM_estimated = {}
test_number = 555

#for k in range(len(sigmoid_slope_array)):
'''
for k in range(0,1):
    CDPR_optimizer.sigmoid_slope = sigmoid_slope_array[k]
    ef_actual,ef_est,ef_feasible,ef_infeasible,ef_total,cm_actual,cm_est,cm_total_actual,cm_total_est = CDPR_optimizer.generate_workspace()
    

    BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/CDPR_test_results/"
    file_name = 'workspace_capacity_margin_'+str(CDPR_optimizer.sigmoid_slope) + str('_') + str(test_number)

    savez(os.path.join(BASE_PATH, file_name),cm_total_actual =cm_total_actual,cm_total_est =cm_total_est,cm_actual =cm_actual,cm_est =cm_est,ef_total=ef_total,\
        ef_actual = ef_actual,ef_est  = ef_est, ef_feasible = ef_feasible, ef_infeasible  = ef_infeasible ,sigmoid_slope_inp = CDPR_optimizer.sigmoid_slope,\
                    base_points  = base_points , pos_bounds = CDPR_optimizer.pos_bounds,desired_vertices  = CDPR_optimizer.cartesian_desired_vertices, \
                        cartesian_dof_input = CDPR_optimizer.cartesian_dof_input, height_params = CDPR_optimizer.height_params , \
                            length_params = CDPR_optimizer.length_params, qdot_min  = CDPR_optimizer.qdot_min, qdot_max = CDPR_optimizer.qdot_max)
    
'''
for k in range(len(sigmoid_slope_array)):
    CDPR_optimizer.sigmoid_slope = sigmoid_slope_array[k]
    BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/CDPR_test_results/"
    #test_number = 5
    file_name = 'workspace_capacity_margin_'+str(CDPR_optimizer.sigmoid_slope) + str('_') + str(test_number)
    data = load(os.path.join(BASE_PATH, file_name)+str('.npz'))
    #file_name = 'workspace_capacity_margin_'+str(CDPR_optimizer.sigmoid_slope) + str('_') + str(test_number))

    ef_total = data['ef_total']
    ef_actual[k] = data['ef_actual']
    ef_estimated_array[k] = data['ef_est']


    
    

    ef_feasible = data['ef_feasible']
    ef_infeasible = data['ef_infeasible']
    CM_total = data['cm_total_actual']
    CM_estimated[k] = data['cm_total_est']


#print('ef_actual',ef_actual)
#print('ef_total',ef_total)
#print('shape of ef_total',shape(ef_total))
#print('shape of ef_actual',shape(ef_actual))


#print('cm_total',CM_total)
#print('shape of cm',shape(CM_total))
pos_CM_total = check_ndarray(CM_total)

#pos_CM_total = pos_CM_total[pos_CM_total > -CDPR_optimizer.lower_bound] ### Changing this i dont know if this will work

pos_CM_total = pos_CM_total[pos_CM_total > -CDPR_optimizer.lower_bound] 
pos_CM_total = sort(pos_CM_total)
#print('pos_CM_total',pos_CM_total)


#print('min of CM',min(pos_CM_total))

#print('pos_CM_total',pos_CM_total)
cm_est_density = {}
cm_est_density_len = {}
cm_est_density[0] = check_ndarray(CM_estimated[0])
cm_est_density_len[0] = len(cm_est_density[0][cm_est_density[0][:] > 0])

cm_est_density[1] = check_ndarray(CM_estimated[1])
cm_est_density_len[1] = len(cm_est_density[1][cm_est_density[1][:] > 0])


cm_est_density[2] = check_ndarray(CM_estimated[2])
cm_est_density_len[2] = len(cm_est_density[2][cm_est_density[2][:] > 0])


cm_est_density[3] = check_ndarray(CM_estimated[3])
cm_est_density_len[3] = len(cm_est_density[3][cm_est_density[3][:] > 0])


cm_est_density[4] = check_ndarray(CM_estimated[4])
cm_est_density_len[4] = len(cm_est_density[4][cm_est_density[4][:] > 0])


cm_est_density[5] = check_ndarray(CM_estimated[5])
cm_est_density_len[5] = len(cm_est_density[5][cm_est_density[5][:] > 0])




cm_total_density = check_ndarray(CM_total)
cm_act_density_len = len(cm_total_density[cm_total_density[:] > 0])




print('actual density is',cm_act_density_len)
print('estimated length',cm_est_density_len)


#cm_ratios = cm_est_density_len*(cm_act_density_len**(-1))
print('ratios are',cm_est_density_len[0]*(cm_act_density_len**(-1) ))
print('ratios are',cm_est_density_len[1]*(cm_act_density_len**(-1) ))
print('ratios are',cm_est_density_len[2]*(cm_act_density_len**(-1) ))
print('ratios are',cm_est_density_len[3]*(cm_act_density_len**(-1) ))
print('ratios are',cm_est_density_len[4]*(cm_act_density_len**(-1) ))
print('ratios are',cm_est_density_len[5]*(cm_act_density_len**(-1) ))

input('stop and final')
cm_est = check_ndarray(CM_estimated[0])
pos_CM_est = cm_est[cm_est>-CDPR_optimizer.lower_bound]

pos_CM_est = sort(pos_CM_est)
#print('pos_CM_est',pos_CM_est)
#for i in range(len(ef_total)):
#for j in range(len(ef_total)):
cm_est = check_ndarray(CM_estimated[1])
pos_CM_est = cm_est[cm_est>-CDPR_optimizer.lower_bound]

pos_CM_est = sort(pos_CM_est)
#print('pos_CM_est 1',pos_CM_est)


cm_est = check_ndarray(CM_estimated[2])
pos_CM_est = cm_est[cm_est>-CDPR_optimizer.lower_bound]

pos_CM_est = sort(pos_CM_est)
#print('pos_CM_est 2',pos_CM_est)

#print('ef_actual',ef_actual)

#q_actual = ef_actual[ef_actual[:,:,0] >= 0]

#q_actual = ef_actual[ef_actual[:,:,0] != -1000]
q_feasible  = ef_feasible[ef_feasible[:,:,0] >= 0]
#q_feasible  = ef_feasible[ef_feasible[:,:,0] != -1000]
#q_feasible = ef_feasible
q_infeasible  = ef_infeasible[ef_infeasible[:,:,0] >= 0]

#q_infeasible  = ef_infeasible[ef_infeasible[:,:,0] != -1000]


#print('q_actual',q_actual)
#MaskedArray()
#print('q_actual is',q_actual)
#print('q_feasible is',q_feasible)
#print('q_infeasible is',q_infeasible)


#print('shape(q_feasible)',shape(q_feasible))
#print('q_estimated is',q_estimated)
#print('shape(q_est)',shape(q_estimated))

#global figure 7

fig_1 = plt.figure()

ax = plt.axes(projection='3d')

CM_estimated_density = CM_estimated[3]

print('shape(CM_estimated density)',shape(CM_estimated_density))
print('shape of ef_total',shape(ef_total[:,:,0]))

#X_dens,Y_dens = meshgrid(ef_total[:,:,0],ef_total[:,:,1])
w = ax.plot_surface(ef_total[:,:,0], ef_total[:,:,1], CM_estimated_density,cmap='spring',alpha=0.4)
# change the fontsize

ax.set_xlabel('x [m]',fontsize=13)
ax.set_ylabel('y [m]',fontsize=13)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#ax.set_zlabel('$\hat{\gamma}$',fontsize=20)
#ax.set_zlabel('$\hat{\gamma}$',rotation=145)
ax.set_zlabel(r"$\hat{\gamma}$" + str(' [N]'),rotation=90,fontsize=16)






#w = ax.plot_surface(ef_total[:,:,0], ef_total[:,:,1], CM_estimated_density,cmap='spring',alpha=0.4)
        # change the fontsize


ax.set_xlabel('x [m]',fontsize=13)
ax.set_ylabel('y [m]',fontsize=13)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#ax.set_zlabel('$\hat{\gamma}$',fontsize=20)
#ax.set_zlabel('$\hat{\gamma}$',rotation=145)
ax.set_zlabel(r"$\hat{\gamma}$" + str(' [N]'),rotation=90,fontsize=16)

### Test gradient descent here
fig_2 = plt.figure()

ax2 = plt.axes()
num_iterations = 1000
learning_rate = 0.0001
### Good starting point x0 = 0.65, y0 = 0.72
#x0 = 0.65
#y0 = 0.72

x0 = 0.005

#x0 = 0.48
y0 = 0.005
y0 = 0.29
#q = array([x_0,y_0])
first_iteration = True

x0_start = array([0.005,0.5,0.9])
y0_start = array([0.1,0.1,0.9])

color_arr = ['cyan','k','green',]
color_arr = ['magenta','k','green','orange']
color_arr = ['magenta','k','green','orange']

color_arr = ['magenta','k','green','k']
color_arr = ['cyan','k','green',]

x_in = x0
y_in = y0



i0_plot = zeros(shape=(num_iterations))
#len(x0_start)
sigmoid_slope_arr = [2.0,5.0,10.0,20.0,30.0,50.0]
error_plot_a = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
error_plot_n = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
z0_plot = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
x0_plot = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))
y0_plot = zeros(shape=(num_iterations,len(sigmoid_slope_arr)))


for lm in range(len(x0_start)):
#for lm in range(1,2):

#for lm in range(3,len(sigmoid_slope_arr)):
    CDPR_optimizer.sigmoid_slope = 50.0
    x0 = x0_start[lm]
    #x0 = 0.45 Good configuration to show
    y0 = y0_start[lm]

    #y0 = 0.30 Good configuration to show
    #x0 = 0.5
    #y0 = 0.1

    x_in = x0
    y_in = y0
    for i in range(num_iterations):
        
        print('number of iterations',i)
        i0_plot[i] = i
        
        q = array([x0,y0])
        #q = array([x_in,y_in])

        x_in = x0
        y_in = y0
        #y_in += learning_rate

        test_joint = 1

        x0_plot[i,lm] = x_in
        y0_plot[i,lm] = y_in
        #W,W_n, H = get_wrench_matrix(q,self.length_params,self.height_params)
        
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
        W = Wm
        
        
        #W = W_n
        #JE = W
        #print('JE is',JE)
        #print('H is',H)
        #print('Wrench matrix is', J)




        h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot = \
            get_polytope_hyperplane(W,CDPR_optimizer.active_joints,CDPR_optimizer.cartesian_dof_input,CDPR_optimizer.qdot_min,CDPR_optimizer.qdot_max,CDPR_optimizer.sigmoid_slope)
            
        
        Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign = \
        get_capacity_margin(W,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,\
                CDPR_optimizer.active_joints,CDPR_optimizer.cartesian_dof_input,CDPR_optimizer.qdot_min,CDPR_optimizer.qdot_max,CDPR_optimizer.cartesian_desired_vertices,CDPR_optimizer.sigmoid_slope)
        
        Wu,Wn,Hess= get_wrench_matrix(q,CDPR_optimizer.length_params,CDPR_optimizer.height_params)
        z0 = Gamma_min_softmax

        z0_plot[i,lm] = z0

        print('capacity margin is',z0)

        print('x position now is',x0)
        print('y position now is',y0)

        dn_dq = normal_gradient(Hess)            
        #ax.scatter(x0,y0,z0,c=color_arr[k],s=4)   
        #plt.pause(0.001)
        #plt.show()

        
        d_gamma_hat,d_LSE_dq ,d_LSE_dq_arr,d_gamma_max_dq,dn_dq = Gamma_hat_gradient_2D(Wn,Hess,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                p_minus_hat,Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat,\
                CDPR_optimizer.qdot_min,CDPR_optimizer.qdot_max,CDPR_optimizer.cartesian_desired_vertices,CDPR_optimizer.sigmoid_slope)

        #print('Gamma min is',Gamma_min)
        ##print('Gammaa_min softmax is',Gamma_min_softmax)
        #print('d_LSE_dq',d_LSE_dq)
        ##print('d_LSE_dq_arr',d_LSE_dq_arr)
        #print('d_gamma_max_dq',d_gamma_max_dq)
        x0 = x0 - learning_rate*d_gamma_hat[0]
        y0 = y0 - learning_rate*d_gamma_hat[1]
        #y0 += learning_rate
        #z0 = Gamma_min_softmax
        
        d_G_dq_a = -d_gamma_hat[test_joint]
        
        #dvk_dq_a = Hess[test_joint,2,:]

        #dvk_dq = Wn[:,2]
        #dn_dq_a = normal_gradient(Hess)

        #d_h_plus_dq_a = d_h_plus_dq
        #d_h_minus_dq_a = d_h_minus_dq
        
        #x_term = matmul(transpose(n_k[2,:]),Wn[:,0])

        #print('x_term is',x_term)

        #dx_dq_a = matmul(dn_dq[test_joint,2,:],Wn[:,0]) + matmul(transpose(n_k[2,:]),Hess[test_joint,0,:])

        

        #dsig_dq_a = sigmoid_gradient(2,0,dn_dq,n_k,Wn,Hess,test_joint,self.sigmoid_slope)
        #dsig_dq = sigmoid(x_term,self.sigmoid_slope)

        #dsig_dq_a = (dsig_dq*(1.0-dsig_dq))*dx_dq_a
        
            
        if not first_iteration:
            #ax2.scatter(i,z0,c='k',s=10)
            #ax.scatter(x0,y0,z0,c='k',s=20)
            d_G_dq_n = (Gamma_min_softmax-prev_Gamma_min_softmax)/(1.0*learning_rate)

            #error_plot[i,lm] = ((d_G_dq_n - d_G_dq_a)/(d_G_dq_n))*100.0

            #error_plot[i,lm] = ((d_G_dq_n - d_G_dq_a))

            #error_plot_a[i,lm] = d_G_dq_a

            #error_plot_n[i,lm] = d_G_dq_n

            error_plot_n[i,lm] = ((d_G_dq_n - d_G_dq_a)/(d_G_dq_n))*100.0
            #print('NUmerical gradient of gamma',(Gamma_min_softmax-prev_Gamma_min_softmax)/(1.0*learning_rate))
            #print('ANalytical gradient',-d_gamma_hat)
            prev_Gamma_min_softmax = Gamma_min_softmax
            #input('test gradient')                    
            #plt.pause(0.0001)
            
            #prev_n_k = n_k
            #prev_h_plus = h_plus_hat
            #prev_h_minus = h_minus_hat
        if first_iteration:
            prev_Gamma_min_softmax = Gamma_min_softmax
            #prev_n_k = n_k
            #prev_h_plus = h_plus_hat
            #prev_h_minus = h_minus_hat
            #dsig_dq_prev = dsig_dq
            #dvk_dq_prev = dvk_dq
            #x_term_prev = x_term
            first_iteration = False
        #input('test error here')


ax.plot(x0_plot[:,0],y0_plot[:,0],z0_plot[:,0],color=color_arr[0],marker='+', linestyle='dashed',label='x0:0.005, y0:0.1 ')
ax.plot(x0_plot[:,1],y0_plot[:,1],z0_plot[:,1],color=color_arr[1],marker='+', linestyle='dashed',label='x0:0.5, y0:0.1 ')
ax.plot(x0_plot[:,2],y0_plot[:,2],z0_plot[:,2],color=color_arr[2],marker='+', linestyle='dashed',label='x0:0.9, y0:0.9 ')
#ax.plot(x0_plot[:,3],y0_plot[:,3],z0_plot[:,3],color=color_arr[3],marker='+', linestyle='dashed')
ax.legend(loc="upper right",fontsize=25)
ax2.plot(i0_plot,z0_plot[:,0],color=color_arr[0],linestyle='dashed',label='x0:0.005, y0:0.1 ')
ax2.plot(i0_plot,z0_plot[:,1],color=color_arr[1],linestyle='dashed',label='x0:0.5, y0:0.1 ')
ax2.plot(i0_plot,z0_plot[:,2],color=color_arr[2],linestyle='dashed',label='x0:0.9, y0:0.9 ')


#ax2.plot(i0_plot,z0_plot[:,3],color=color_arr[3],linestyle='dashed')
ax2.set_xlabel('Number of Iterations',fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
#ax2.set_xlabel('Number of Iterations',prop={'size': 20})
ax2.set_ylabel(r"$\hat{\gamma}$" + str(' [N]'),fontsize=25)
ax2.legend(loc="lower right",fontsize=25)

'''

ax2.plot(y0_plot[1:,0],error_plot_n[1:,0],color=color_arr[0],linestyle='solid',label='Numerical Slope: 10')
ax2.plot(y0_plot[1:,0],error_plot_n[1:,1],color=color_arr[1],linestyle='solid',label='Numerical Slope: 20')
ax2.plot(y0_plot[1:,0],error_plot_n[1:,2],color=color_arr[2],linestyle='solid',label='Numerical Slope: 30')
ax2.plot(y0_plot[1:,0],error_plot_n[1:,3],color=color_arr[3],linestyle='solid',label='Numerical Slope: 50')
ax2.set_xlabel('Y (m)')
#ax2.set_ylabel(r"$\partial \hat{\gamma}$",fontsize=15)
ax2.set_ylabel('Error $(\%)$',fontsize=15)
'''
#plt.legend(loc="lower left")
#plt.tight_layout()
plt.show()





## Stopping here

color_array = ['k','white','brown','darkorange','magenta','blue','white','magenta','blue']
plt_estimate = {}

boundary_estimate = {}
boundary_actual = {}

fig_2 = plt.figure()

ef = array([0.25,0.75])


for i in range(len(base_points)):
    x_plt = array([ef[0],base_points[i,0]])
    y_plt = array([ef[1],base_points[i,1]])
    plt.plot(x_plt,y_plt,color='cyan')

plt_feasible = plt.scatter(q_feasible[:,0], q_feasible[:,1],color = 'g',s=0.0005)
plt_infeasible = plt.scatter(q_infeasible[:,0], q_infeasible[:,1],color = 'r',s=0.0005)




for k in range(len(sigmoid_slope_array)):

    if (k<7.0):

        ef_estimated = ef_estimated_array[k]
        q_estimated = ef_estimated[ef_estimated[:,:,0] >= 0]
        ef_actual_arr = ef_actual[k]
        q_actual_new = ef_actual_arr[ef_actual_arr[:,:,0] >= 0]
        
        

        
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
#plt.xlim(-0.02,1.02)
#plt.ylim(-0.02,1.02)
#plt.savefig()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig('Wrench Feasible Workspace_' + str(CDPR_optimizer.sigmoid_slope)+('.png'),dpi=600)




















plt_empty = plt.scatter(ef[0],ef[1],color='g',s=0.0000001)

plt_feasible = plt.scatter(q_feasible[:,0], q_feasible[:,1],color = 'g',s=0.1001)
plt_infeasible = plt.scatter(q_infeasible[:,0], q_infeasible[:,1],color = 'r',s=0.1001)


q_actual = q_actual_new

for k in range(len(sigmoid_slope_array)):

    if (k<7):

        ef_estimated = ef_estimated_array[k]
        q_estimated = ef_estimated[ef_estimated[:,:,0] >= 0]
        print('q_estimated',q_estimated)

        if (k == 0):
            print('sgimopid 1 is',q_estimated)
            input('stop to test')

        #q_actual_sort = argsort(q_actual,axis=0)
        #q_estimated_sort = argsort(q_estimated,axis=0)
        #plt_actual = plt.plot(q_actual[:,0], q_actual[:,1],color = 'k')
        
        #plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],color = 'k',s=2.0)
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')

        
        print('q_actual is',q_actual)
        boundary_actual_a = Polygon(q_actual)
        #print('q_actual is',shape(q_actual_new))
        
        #print('q_estimated',q_estimated)

        #q_actual_sort = argsort(q_actual,axis=0)
        #q_estimated_sort = argsort(q_estimated,axis=0)
        #plt_actual = plt.plot(q_actual[:,0], q_actual[:,1],color = 'k')
        
        #plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],color = 'k',s=2.0)
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        #plt_estimate = plt.plot(q_estimated[:,0], q_estimated[:,1],color = 'cyan')
        print('len(q_actual)',len(q_actual_new))
        print('len(q_estimated)',len(q_estimated))
        print('Ratio is',len(q_estimated)*(len(q_actual_new))**(-1))
        

        boundary_estimate[k] = Polygon(q_estimated)
        closest_points = nearest_points(boundary_estimate[k], boundary_actual_a)


        print('closest points 1 are',closest_points[0].y)
        print('closest points 2 are',closest_points[1].y)

        x1 = closest_points[0].x
        x2 = closest_points[1].x


        y1 = closest_points[0].y
        y2 = closest_points[1].y


        print('closest points 1 are',closest_points[0].x)
        print('closest points 2 are',closest_points[1].x)

        plt.scatter(x1,y1,color='y',s=100)
        plt.scatter(x2,y2,color='y',s=100)

        d_dh = directed_hausdorff(q_estimated,q_actual_new)[0]

        d_dh = directed_hausdorff(q_actual_new, q_estimated)[0]


        print('directed hausdorff',d_dh)


        distance = closest_points[0].distance(closest_points[1]) 
        print('distance is',distance)

        if boundary_estimate[k].intersects(boundary_actual_a):
            print("The boundaries intersect.")
        else:
            print("The boundaries do not intersect.")
        plt_estimate[k] = plt.scatter(q_estimated[:,0], q_estimated[:,1],facecolor = color_array[k],marker='o',s=5.0)
        input('record this')
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


#plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],facecolors='none', edgecolors='k',marker = 'D',alpha=0.75,s=0.150)
plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],facecolors='none', edgecolors='k',marker = 'D',alpha=0.75,s=18)

plt.legend((plt_feasible,plt_infeasible,plt_empty,plt_actual,plt_empty,plt_empty ,plt_estimate[0],plt_estimate[1],plt_estimate[2],plt_estimate[3],plt_estimate[4],plt_estimate[5]),\
    ('WFW','Wrench Infeasbible', '','Actual Boundary','','Estimated Boundary','Slope-' + str(sigmoid_slope_array[0]),\
        'Slope-' + str(sigmoid_slope_array[1]),'Slope-' + str(sigmoid_slope_array[2]),\
           'Slope-' + str(sigmoid_slope_array[3]),'Slope-' + str(sigmoid_slope_array[4]),
           'Slope-' + str(sigmoid_slope_array[5]))\
                ,markerscale=4,loc=3)

plt.savefig('Wrench Feasible Workspace full_' + str(CDPR_optimizer.sigmoid_slope)+('_')+str(test_number)+('.png'),dpi=600)
plt.show()




### Gradient plot 



