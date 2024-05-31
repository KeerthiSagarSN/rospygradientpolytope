#!/usr/bin/env python
## Library import
############# ROS Dependencies #####################################
import rospy
import os
from geometry_msgs import msg
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped,WrenchStamped, PointStamped
from std_msgs.msg import Bool, Float32


from sensor_msgs.msg import Joy, JointState, PointCloud 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial import distance as dist_scipy

import tf_conversions as tf_c
from tf2_ros import TransformBroadcaster


#### Polytope Plot - Dependency #####################################

## Polygon plot for ROS - Geometry message
from jsk_recognition_msgs.msg import PolygonArray, SegmentArray
from geometry_msgs.msg import Polygon, PolygonStamped, Point32, Pose
import time





import time
from numpy.linalg import det
from numpy import sum,mean,average,inf,arange,mean
import matplotlib.pyplot as plt


BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/test_results/"


test_case = 1

file_name = 'error_plot_UR_'+str(test_case)

#################### Linear Algebra ####################################################

from numpy.core.numeric import cross
from numpy import matrix,matmul,transpose,isclose,array,rad2deg,abs,vstack,hstack,shape,eye,zeros,random,savez,load
from numpy import polyfit,poly1d,count_nonzero

from numpy import float64,average,clip

from numpy.linalg import norm,det
from math import atan2, pi,asin,acos

sigmoid_slope_test = array([50,100,150,200,400])

'''
savez(os.path.join(BASE_PATH, file_name),q_in = self.q_in_array,sigmoid_slope_inp = sigmoid_slope_test, ts = self.ts_arr,\
    Gamma_min_array = self.Gamma_min_array, Gamma_min_softmax_array = self.Gamma_min_softmax_array,Error_gamma_array = self.Error_gamma_array)
'''

sigmoid_slope_plot = ['50', '100', '150',
        '200', '400']

data_load = load(os.path.join(BASE_PATH, file_name)+str('.npz'))
Error_gamma_array_UR = data_load['Error_gamma_array']
## Get the maximum value of the Capacity margin to normalize the vector of error
Gamma_min_array_UR = data_load['Gamma_min_array']

max_gamma_UR = zeros(shape=(len(sigmoid_slope_test)))

for i in range(len(sigmoid_slope_test)):
    max_gamma_UR[i] = max(Gamma_min_array_UR[:,i])



test_case = 3

BASE_PATH = "/home/imr/catkin_ws_build/src/rospygradientpolytope/test_results/"


file_name = 'error_plot_sawyer_'+str(test_case)



data_load = load(os.path.join(BASE_PATH, file_name)+str('.npz'))
Error_gamma_array = data_load['Error_gamma_array']
Gamma_min_array = data_load['Gamma_min_array']


## Get the maximum value of the Capacity margin to normalize the vector of error
max_gamma = zeros(shape=(len(sigmoid_slope_test)))

for i in range(len(sigmoid_slope_test)):
    max_gamma[i] = max(Gamma_min_array[:,i])
#max_gamma = data_load['max_gamma']



fig, ax = plt.subplots()

#max_gamma = zeros(shape=(len(sigmoid_slope_test)))

## Get the maximum value of the Capacity margin to normalize the vector of error



data_to_plot = [Error_gamma_array[:,0]/(1.0*max_gamma[0]),\
                Error_gamma_array[:,1]/(1.0*max_gamma[1]),\
                    Error_gamma_array[:,2]/(1.0*max_gamma[2]),\
                        Error_gamma_array[:,3]/(1.0*max_gamma[3]),
                        Error_gamma_array[:,4]/(1.0*max_gamma[4])]

data_to_plot_UR = [Error_gamma_array_UR[:,0]/(1.0*max_gamma_UR[0]),\
                Error_gamma_array_UR[:,1]/(1.0*max_gamma_UR[1]),\
                    Error_gamma_array_UR[:,2]/(1.0*max_gamma_UR[2]),\
                        Error_gamma_array_UR[:,3]/(1.0*max_gamma_UR[3]),
                        Error_gamma_array_UR[:,4]/(1.0*max_gamma_UR[4])]

# List of labels from sigmoid slope
ax.set_xticklabels(['','50','100','150','200','400'])
plt.xlabel("Sigmoid Slope",size=15)
plt.ylabel("Error",size=15)
# Create the boxplot
#bp = ax.violinplot(data_to_plot)
counter = 0
color_arr = ['orange','blue']

# Create the boxplot
#bp = ax.violinplot(data_to_plot)


bp = ax.violinplot(data_to_plot, points=100, positions=arange(0, len(data_to_plot)),
               showmeans=False, showextrema=False, showmedians=False)


for b in bp['bodies']:
    #b.set_facecolor(color_arr[counter])
    m = mean(b.get_paths()[0].vertices[:, 0])

    b.set_facecolor('orange')

    b.set_edgecolor('orange')
    b.get_paths()[0].vertices[:, 0] = clip(b.get_paths()[0].vertices[:, 0], -inf, m)
    #b.set_color('orange')
    
#ax.set_aspect(1)

#plt.show()


counter += 1




#bp2 = ax.violinplot(data_to_plot_UR)


bp2 = ax.violinplot(data_to_plot_UR, points=100, positions=arange(0, len(data_to_plot_UR)),
               showmeans=False, showextrema=False, showmedians=False)


for b2 in bp2['bodies']:
    b2.set_facecolor('blue')
    b2.set_edgecolor('blue')
    #b2.get_paths()[0].vertices[:] = clip(b2.get_paths()[0].vertices[:], -inf)
    m = mean(b2.get_paths()[0].vertices[:, 0])
    b2.get_paths()[0].vertices[:, 0] = clip(b2.get_paths()[0].vertices[:, 0], m, inf)

    #b2.set_color('blue')

ax.legend([bp['bodies'][0],bp2['bodies'][0]],['Sawyer', 'UR5'])

plt.savefig('Error_plot_violin_' + str(test_case)+('.png'))
plt.show()
'''
#fig2, ax2 = plt.subplots()
'''

# List of five airlines to plot


'''

# Violin plot
sns.violinplot(x = sigmoid_slope_plot , y = data_to_plot,\
               hue = data_to_plot_UR,\
               split = True)
'''
# Iterate through the five airlines
'''

for i in range(len(sigmoid_slope_plot)):
    # Subset to the airline
    subset = flights[flights['name'] == airline]
    
    # Draw the density plot
    sns.distplot(subset['arr_delay'], hist = False, kde = True,
                kde_kws = {'linewidth': 3},
                label = sigmoid_slope_plot[i])
plt.legend(prop={'size': 16}, title = 'Sigmoid Slope')
#plt.title('Density Plot with Multiple Airlines')
plt.xlabel('Error (%)')
plt.ylabel('Capacity Margin')

sns.distplot(subset['arr_delay'], hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 3}, 
            label = Sigmoid Slope)
    

# Plot formatting

# Plot formatting

ax2.set_ylim([-0.02, 0.12])
#self.Gamma_min_array[:,0] = self.Gamma_min_array[:,0]

plt_sigmoid_50 = plt.scatter(self.Gamma_min_array[:,0],self.Error_gamma_array[:,0]/(1.0*max_gamma[0]),color='c',s=0.5,alpha=0.5)

#calculate equation for trendline
z_50 = polyfit(self.Gamma_min_array[:,0], self.Error_gamma_array[:,0]/(1.0*max_gamma[0]), 1)
p_50 = poly1d(z_50)

#add trendline to plot
plt.plot(self.Gamma_min_array[:,0], p_50(self.Gamma_min_array[:,0]),color='c')   

plt_sigmoid_100 = plt.scatter(self.Gamma_min_array[:,1],self.Error_gamma_array[:,1]/(1.0*max_gamma[1]),color='m',s=0.5,alpha=0.5)

#calculate equation for trendline
z_100 = polyfit(self.Gamma_min_array[:,1], self.Error_gamma_array[:,1]/(1.0*max_gamma[1]), 1)
p_100 = poly1d(z_100)

#add trendline to plot
plt.plot(self.Gamma_min_array[:,1], p_100(self.Gamma_min_array[:,1]),color='m')   


plt_sigmoid_150 = plt.scatter(self.Gamma_min_array[:,2],self.Error_gamma_array[:,2]/(1.0*max_gamma[2]),color='y',s=0.5,alpha=0.5)


#calculate equation for trendline
z_150 = polyfit(self.Gamma_min_array[:,2], self.Error_gamma_array[:,2]/(1.0*max_gamma[2]), 1)
p_150 = poly1d(z_150)

#add trendline to plot
plt.plot(self.Gamma_min_array[:,2], p_150(self.Gamma_min_array[:,2]),color='y')   

plt_sigmoid_200 = plt.scatter(self.Gamma_min_array[:,3],self.Error_gamma_array[:,3]/(1.0*max_gamma[3]),color='k',s=0.5,alpha=0.5)


#calculate equation for trendline
z_200 = polyfit(self.Gamma_min_array[:,3], self.Error_gamma_array[:,3]/(1.0*max_gamma[3]), 1)
p_200 = poly1d(z_200)

#add trendline to plot
plt.plot(self.Gamma_min_array[:,3], p_200(self.Gamma_min_array[:,3]),color='k')   

plt_sigmoid_400 = plt.scatter(self.Gamma_min_array[:,4],self.Error_gamma_array[:,4]/(1.0*max_gamma[4]),color='r',s=0.5,alpha=0.5)
#plt.scatter(plt_sigmoid_50_x,plt_sigmoid_50_y,'r')


#calculate equation for trendline
z_400 = polyfit(self.Gamma_min_array[:,4], self.Error_gamma_array[:,4]/(1.0*max_gamma[4]), 1)
p_400 = poly1d(z_400)

#add trendline to plot
plt.plot(self.Gamma_min_array[:,4], p_400(self.Gamma_min_array[:,4]),color='r')  

plt.legend((plt_sigmoid_50 ,plt_sigmoid_100 ,plt_sigmoid_150 ,plt_sigmoid_200, plt_sigmoid_400  ),('50','100', '150','200','400'),title='Sigmoid Slope',markerscale=5)
plt.xlabel('Actual Capacity Margin (m/s)',size=15)
plt.ylabel('Error',size = 15,labelpad=-6)
#self.Gamma_min_array

#self.Gamma_min_array



plt.show()







savez(os.path.join(BASE_PATH, file_name),q_in = self.q_in_array,sigmoid_slope_inp = sigmoid_slope_test, ts = self.ts_arr,\
    Gamma_min_array = self.Gamma_min_array, Gamma_min_softmax_array = self.Gamma_min_softmax_array,Error_gamma_array = self.Error_gamma_array)
'''