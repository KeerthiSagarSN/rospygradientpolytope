# ROS - pygradientpolytope

This repository includes nodes for generating and visualizing Cartesian Available : (1) Velocity Polytope (2) Force Polytope (3) Desired Polytope
(4) Performance Index - Capacity Margin in [Rviz](http://wiki.ros.org/rviz).

## Installation
This repository was tested on Universal Robot [UR5](https://www.universal-robots.com/products/ur5-robot/). The robot is controllled through an external PC with [ROS Noetic](http://wiki.ros.org/noetic) and Ubuntu 20.04 LTS. 
### Hardware requirements
* External PC. Our specifications are;
```
Intel® Core™ i7-10700K CPU @ 3.80GHz × 16
Intel® Core™ i9-10700K CPU @ 3.80GHz × 16 (Interactive Inverse Kinematics for the Serial Arm-  video for the paper)
RAM: 16 GB
Graphics: NVIDIA Corporation GP106GL [Quadro P2200]
```


### Software and Library Requirements 

* Ubuntu 20.04 LTS
* ROS Noetic
If you are new to ROS, go [here](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) to learn how to create a catkin workspace. 
* Polytope. This is the source repository for polytope, a toolbox for geometric operations on polytopes in any dimension.
[Polytope](https://pypi.org/project/polytope/).
* [pykdl_utils](http://wiki.ros.org/pykdl_utils) Higher Level Python wrapper for PyKDL in ROS for Kinematic Solver
* Python native libraries [Scipy](https://scipy.org/), [Numpy](https://numpy.org/)
* Polygon ROS geometry messages for plotting in Rviz [jsk-ros-pkg](https://github.com/jsk-ros-pkg/jsk_recognition)
* Polytope ros message publisher forked and modified from Pycapacity Library [capacity_visual_utils](https://github.com/askuric/polytope_vertex_search/blob/master/ROS_nodes/panda_capacity/scripts/capacity/capacity_visual_utils.py)



### Toolbox in the Catkin workspace - ROS1
In a Terminal
```
$ cd ~/catkin_ws/src/
$ git clone https://gitlab.com/KeerthiSagarSN/rospygradientpolytope
```

### Library Installation - Only if not preinstalled

In a terminal
```
$ pip install sympy
$ pip install polytope
```
### Clone Sawyer Robot- Different from the Original Repo due to the offset in the tool center point to avoid Null in Jacobian
```
$ git clone https://github.com/KeerthiSagarSN/sawyer_robot
```

### Clone UR5 Robot Repository - Different from the Original Repo due to the offset in the tool center point to avoid Null in Jacobian

```
$ git clone https://github.com/KeerthiSagarSN/universal_robot
```


### Clone PyKDL - From OROCOS-KDL Repository - Latest Branch

```
$ git clone https://github.com/orocos/orocos_kinematics_dynamics.git
```

### Clone Pykdl - Kinematics Wrapper Repository - Edited for Python 3 and ROS Noetic Compatible
```
$ git clone https://github.com/KeerthiSagarSN/hrl-kdl.git
$ git checkout Noetic-devel
$ sudo apt-get install ros-noetic-urdf-parser-plugin
$ sudo apt-get install ros-noetic-urdfdom-py

```
### Clone Visualization Package Library - Rviz
```
$ git clone https://github.com/jsk-ros-pkg/jsk_visualization.git
```

### Catkin Build/ Catkin Make - Build & Source all repositories in Catkin Workspace
```
$ cd ..
$ catkin build
$ source devel/setup.bash
```

#### IF you encounter CMAKE Error : Install all ROS- Dependencies - (Only when CMAKE Error !)
##### Replace underscores "_" with "-" in the package name while typing in the MISSING-PACKAGE below
```
$ sudo apt install ros-noetic-MISSING-PACKAGE

```

#### ROS - Preliminary Dependencies - Install Only if required

```
$ sudo apt-get install ros-$ROS_DISTRO-robot-state-publisher ros-$ROS_DISTRO-joint-state-publisher
$ sudo apt-get install ros-$ROS_DISTRO-joint-state-publisher-gui
```

### To Visualize UR5 robot in Rviz
```
$ roslaunch ur_description view_ur5.launch 
```


### To Visualize Sawyer Robot in Rviz
```
$ roslaunch sawyer_description test_sawyer_description.launch
```

### Launch the IK Controller
#### UR Robot
```
$ roslaunch rospygradientpolytope launch_robot_ur.launch

```
![UR5 - 6 DOF](./Images_Readme/ur5_rviz_readme.png)

#### or
#### Sawyer Robot
```
$ roslaunch rospygradientpolytope launch_robot_sawyer.launch
```
![Sawyer Robot - 7 DOF](./Images_Readme/sawyer_rviz_readme.png)




### Steps to start Interactive Inverse Kinematics
#### Clone the interactive PyQT GUI repository and launch the QT panel
```
$ git clone https://gitlab.com/KeerthiSagarSN/inverse_kinematics_interactive_rviz.git
$ cd ..
$ catkin build
$ source devel/setup.bash
$ roslaunch inverse_kinematics_interactive_rviz inverse_kinematics_interactive_rviz.launch
```
1. Click and drag the interactive sphere with axes in the Rviz window 
2. Select Run IK button from the Interactive panel
3. Click on Polytope:ON for visualizing the estimated capacity margin during optimization
4. Polytope:Off To stop updating the polytope (Faster execution option)
5. Click on Reset Button : After optimization termination for consecutive IK optimization

##### Interactive panel is in experimental phase and may lead to occasional crash. Please restart the node, when GUI crashes.

#### UR Robot
![UR5 - 6 DOF](./Images_Readme/UR5_IK.gif)

#### Sawyer Robot
![Sawyer- 7 DOF](./Images_Readme/Sawyer_IK.gif)

## Module functions
##### Python3 library functions independent of ROS
###### For computing desired or available/feasible velocity polytope faces and vertices
```
from rospygradientpolytope.visual_polytope import *
```
###### Compute polytope facets and vertices of user-defined desired polytope
```
rospygradientpolytope.visual_polytope.desired_polytope 
```
###### Parameters: 
<b> cartesian_desired_vertices </b> - Desired velocities at the end-effector
###### Return: 
<b> polytope_vertices </b>- Convex hull of the desired cartesian vertices<br> 
<b> polytope_faces </b>- Simplices of the convex hull mapped to polytope facets<br> <hr>


###### Compute polytope facets and vertices of available/feasible polytope, with the closest (vertex,facet) pair
```
rospygradientpolytope.visual_polytope.velocity_polytope_with_estimation 

[polytope_vertices, polytope_faces, facet_pair_idx, capacity_margin_faces, capacity_proj_vertex, polytope_vertices_est, polytope_faces_est, capacity_margin_faces_est, capacity_proj_vertex_est,Gamma_min_softmax] = velocity_polytope_with_estimation(JE, qdot_max, qdot_min,cartesian_desired_vertices, sigmoid_slope)

```
###### Parameters: 
<b> JE </b> - Jacobian of the serial manipulator using PyKDL<br>
<b> qdot_max </b> - maximum joint velocity vector- minimum joint velocity vector<br>
<b> qdot_min </b> - minimum joint velocity vector<br>
<b> cartesian_desired_vertices </b> - Desired velocities at the end-effector<br>
<b> sigmoid_slope </b> - Slope parameter for sigmoid activation<br>
###### Return: 
<b> polytope_vertices </b>- Convex hull of the available cartesian velocities <br>
<b> polytope_faces </b>- Facets of the available cartesian velocities <br>
<b> facet_pair_idx </b>- Index of the closest (vertex,facet) pair<br>
<b> capacity_margin_faces </b>- Closest facet in the available cartesian velocities contributing to classical capacity margin<br>
<b> capacity_proj_vertex </b>- Closest vertex in the desired cartesian velocities contributing to classical capacity margin<br>
<b> polytope_vertices_est </b>- Convex hull of the available cartesian velocities with estimated hyperplane parameters<br>
<b> polytope_faces_est </b>- Facets of the available cartesian velocities with estimated hyperplane parameters<br>
<b> capacity_margin_faces_est </b>- Closest facet in the available cartesian velocities contributing to estimated capacity margin<br>
<b> capacity_proj_vertex_est </b>- Closest vertex in the desired cartesian velocities contributing to estimated capacity margin<br>
<b> Gamma_min_softmax </b>- Estimated capacity margin (Performance measure)<br><hr>

###### For computing all hyperplane parameters and estimated capacity margin
```
from rospygradientpolytope.polytope_functions import *
```
###### Compute hyperplane parameters for velocity polytope
```
rospygradientpolytope.polytope_functions.get_polytope_hyperplane

[h_plus,h_plus_hat,h_minus,h_minus_hat,p_plus,p_minus,p_plus_hat,p_minus_hat,n_k, Nmatrix, Nnot] = get_polytope_hyperplane(JE,active_joints,cartesian_dof_input,qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope)
```
###### Parameters: 
<b> JE </b> - Jacobian of the serial manipulator using PyKDL<br>
<b> active_joints </b> - Number of joints in the serial manipulator (Eg, for Sawyer,active_joints = 7) <br>
<b> cartesian_dof_input </b> - Cartesian DOF (Eg, Cartesian DOF = 3), if only translational velocities are considered<br>
<b> qdot_min </b> - minimum joint velocity vector<br>
<b> qdot_max </b> - maximum joint velocity vector- minimum joint velocity vector<br>
<b> cartesian_desired_vertices </b> - Desired velocities at the end-effector<br>
<b> sigmoid_slope </b> - Slope parameter for sigmoid activation<br>
###### Return: 
<b> h_plus </b>- Maximum distance vector contributing to defining available velocities hyperplane   <br>
<b> h_plus_hat </b>- Maximum distance vector contributing to defining available velocities hyperplane (using proposed method) <br>
<b> h_minus </b>- Minimum distance vector contributing to defining available velocities hyperplane   <br>
<b> h_minus_hat </b>- Minimum distance vector contributing to defining available velocities hyperplane (using proposed method) <br>
<b> p_plus </b>- Points on the hyperplane defined by <b>h_plus</b><br>
<b> p_minus </b>- Points on the hyperplane defined by <b>h_minus</b><br>
<b> p_plus_hat </b>- Points on the hyperplane defined by <b>h_plus_hat</b><br>
<b> p_minus_hat </b>- Points on the hyperplane defined by <b>h_minus_hat</b><br>
<b> n_k </b>- Normal vector corresponding to the cross product between two joint twist<br>
<b> Nmatrix </b>- Combinations of twists to define <b>n_k</b><br>
<b> Nnot </b>- Combinations of twists to define <b>h_plus</b>, <b>h_minus</b> using <b>Nmatrix</b> and <b>n_k</b><br><hr>

###### Compute classical and estimated capacity margin (performance measure)
```
rospygradientpolytope.polytope_functions.get_capacity_margin

[Gamma_minus, Gamma_plus, Gamma_total_hat, Gamma_min, Gamma_min_softmax, Gamma_min_index_hat, facet_pair_idx, hyper_plane_sign] = get_capacity_margin(JE,n_k,h_plus,h_plus_hat,h_minus,h_minus_hat,active_joints,cartesian_dof_input,qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope)
```
###### Parameters: 
<b> JE </b>- Jacobian of the serial manipulator using PyKDL<br>
<b> n_k </b>- Normal vector corresponding to the cross product between two joint twist<br>
<b> h_plus </b>- Maximum distance vector contributing to defining available velocities hyperplane   <br>
<b> h_plus_hat </b>- Maximum distance vector contributing to defining available velocities hyperplane (using proposed method) <br>
<b> h_minus </b>- Minimum distance vector contributing to defining available velocities hyperplane   <br>
<b> h_minus_hat </b>- Minimum distance vector contributing to defining available velocities hyperplane (using proposed method) <br>
<b> active_joints </b> - Number of joints in the serial manipulator (Eg, for Sawyer,active_joints = 7) <br>
<b> cartesian_dof_input </b> - Cartesian DOF (Eg, Cartesian DOF = 3), if only translational velocities are considered<br>
<b> qdot_min </b> - minimum joint velocity vector<br>
<b> qdot_max </b> - maximum joint velocity vector- minimum joint velocity vector<br>
<b> cartesian_desired_vertices </b> - Desired velocities at the end-effector<br>
<b> sigmoid_slope </b> - Slope parameter for sigmoid activation<br>
###### Return: 
<b> Gamma_minus </b>- Distance vector contributing to capacity margin using points <b>p_minus</b>  on the hyperplane defined by <b>h_minus</b> <br>
<b> Gamma_plus </b>- Distance vector contributing to capacity margin using points <b>p_plus</b>  on the hyperplane defined by <b>h_plus</b> <br>
<b> Gamma_total_hat </b>- Total distance vector contributing to estimated capacity margin   <br>
<b> Gamma_min </b>- Classical capacity margin <br>
<b> Gamma_min_softmax </b>- Estimated capacity margin (proposed performance measure)<br>
<b> Gamma_min_index_hat </b>- Index of Gamma_total_hat with the minimum value <br>
<b> facet_pair_idx </b>- Index of the (vertex,facet) pair to plot in rviz <br>
<b> hyper_plane_sign </b>- To identify the hyperplane closer to the vertex to plot in rviz <br><hr>

###### For computing analytical gradient of hyperplane parameters and estimated capacity margin for serial robot
```
from rospygradientpolytope.polytope_gradient_functions import *
```

###### Compute hyperplane gradient using the proposed method
```
rospygradientpolytope.polytope_gradient_functions.hyperplane_gradient

[d_h_plus_dq, d_h_minus_dq, dn_dq] = hyperplane_gradient(JE,H,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,p_minus_hat,qdot_min,qdot_max,test_joint,sigmoid_slope)
```
###### Parameters: 
<b> JE </b>- Jacobian of the serial manipulator using PyKDL<br>
<b> H </b>- Hessian of the serial manipulator<br>
<b> n_k </b>- Normal vector corresponding to the cross product between two joint twist<br>
<b> Nmatrix </b>- Combinations of twists to define <b>n_k</b><br>
<b> Nnot </b>- Combinations of twists to define <b>h_plus</b>, <b>h_minus</b> using <b>Nmatrix</b> and <b>n_k</b><br>
<b> h_plus_hat </b>- Maximum distance vector contributing to defining available velocities hyperplane (using proposed method) <br>
<b> h_minus_hat </b>- Minimum distance vector contributing to defining available velocities hyperplane (using proposed method) <br>
<b> p_plus_hat </b>- Points on the hyperplane defined by <b>h_plus_hat</b><br>
<b> p_minus_hat </b>- Points on the hyperplane defined by <b>h_minus_hat</b><br>
<b> qdot_min </b> - minimum joint velocity vector<br>
<b> qdot_max </b> - maximum joint velocity vector- minimum joint velocity vector<br>
<b> test_joint </b> - Gradient with respect to the corresponding joint<br>
<b> sigmoid_slope </b> - Slope parameter for sigmoid activation<br>
###### Return: 
<b> d_h_plus_dq </b>- Gradient of the h_plus_hat vector with respect to the corresponding joint <b> q </b> <br>
<b> d_h_minus_dq </b>- Gradient of the h_minus_hat vector with respect to the corresponding joint <b> q </b> <br>
<b> dn_dq </b>- Gradient of n_k vector with respect to the corresponding joint <b> q </b> <br><hr>

##### Compute estimated capacity margin (performance index) gradient using the proposed method
##### Using multi-processing toolbox with global variable jac_output- representing the analytical gradient of the estimated capacity margin with respect to all active joints in the serial manipulator
```
rospygradientpolytope.polytope_gradient_functions.Gamma_hat_gradient_dq

[jac_output] = Gamma_hat_gradient_dq(JE,H,n_k,Nmatrix, Nnot,h_plus_hat,h_minus_hat,p_plus_hat,\
                        p_minus_hat, Gamma_total_hat, Gamma_min_index_hat,\
                        qdot_min,qdot_max,cartesian_desired_vertices,sigmoid_slope,test_joint,jac_output)
```
###### Parameters: 
<b> JE </b>- Jacobian of the serial manipulator using PyKDL<br>
<b> H </b>- Hessian of the serial manipulator<br>
<b> n_k </b>- Normal vector corresponding to the cross product between two joint twist<br>
<b> Nmatrix </b>- Combinations of twists to define <b>n_k</b><br>
<b> Nnot </b>- Combinations of twists to define <b>h_plus</b>, <b>h_minus</b> using <b>Nmatrix</b> and <b>n_k</b><br>
<b> h_plus_hat </b>- Maximum distance vector contributing to defining available velocities hyperplane (using proposed method) <br>
<b> h_minus_hat </b>- Minimum distance vector contributing to defining available velocities hyperplane (using proposed method) <br>
<b> p_plus_hat </b>- Points on the hyperplane defined by <b>h_plus_hat</b><br>
<b> p_minus_hat </b>- Points on the hyperplane defined by <b>h_minus_hat</b><br>
<b> Gamma_total_hat </b>- Total distance vector contributing to estimated capacity margin   <br>
<b> qdot_min </b> - minimum joint velocity vector<br>
<b> qdot_max </b> - maximum joint velocity vector- minimum joint velocity vector<br>
<b> cartesian_desired_vertices </b> - Desired velocities at the end-effector<br>
<b> sigmoid_slope </b> - Slope parameter for sigmoid activation<br>
<b> test_joint </b> - Gradient with respect to the corresponding joint<br>
<b> jac_output </b> - Global multiprocessing array for storing analytical gradient of the estimated capacity margin <br>
###### Return: 
<b> jac_output</b>- Analytical gradient of the estimated capacity margin with respect to the corresponding test joint <b> test_joint </b> <br><hr>

###### Custom ROS Message to plot polytope,vertices and facets in Rviz
```
from rospygradientpolytope.polytope_ros_message import *
```

### To test capacity margin gradient (Sawyer Robot)
Fix all joints and move only one-joint "test_joint" and visualize simultaneously numerical gradient and analytical gradient 
In launch_robot_sawyer.py --> uncomment self.test_gamma_gradient(sigmoid_slope_test=sigmoid_slope) in __init__
```
$ roslaunch rospygradientpolytope launch_robot_sawyer.launch
```

###### To test capacity margin gradient (UR5 Robot)
Fix all joints and move only one-joint "test_joint" and visualize simultaneously numerical gradient and analytical gradient 
In launch_robot_ur.py --> uncomment self.test_gamma_gradient(sigmoid_slope_test=sigmoid_slope) in __init__
```
$ roslaunch rospygradientpolytope launch_robot_ur.launch
```
#### Cable Driven Parallel Robots (CDPR)- Planar
##### To generate workspace of the CDPR with different sigmoid slopes (4-cable, 2-DOF)
Generate Wrench Feasible Workspace (WFW) by identifying where Capacity margin index = 0
```
$ python3 CDPR_workspace.py
```

###### Simple First-order Gradient Descent Optimization (CDPR)
###### Start from initial pose and use analytical gradient to reach pose with maximum capacity margin
```
$ python3 CDPR_workspace.py
```
### Region-of-Interest Optimization (CDPR)
###### Optimization to operate CDPR within a ROI with obstacle avoidance
Plotting with matplotlib is slow in the current implementation, optimized code to plot is under development
```
$ python3 test_CDPR_Optimization.py
```


# References

If you found this code useful, we would be grateful if you cite the following reference:

[1] K. Sagar, S. Caro, T. Padır and P. Long, "Polytope-Based Continuous Scalar Performance Measure With Analytical Gradient for Effective Robot Manipulation," in IEEE Robotics and Automation Letters, vol. 8, no. 11, pp. 7289-7296, Nov. 2023, doi: 10.1109/LRA.2023.3313926.

Bibtex:

```
@ARTICLE{10246372,
  author={Sagar, Keerthi and Caro, Stéphane and Padır, Taşkın and Long, Philip},
  journal={IEEE Robotics and Automation Letters}, 
  title={Polytope-Based Continuous Scalar Performance Measure With Analytical Gradient for Effective Robot Manipulation}, 
  year={2023},
  volume={8},
  number={11},
  pages={7289-7296},
  doi={10.1109/LRA.2023.3313926}}
```

You can find the video accompanying for the article [here](https://ieeexplore.ieee.org/abstract/document/10246372/media#media)

Youtube link for the video can be found [here](https://www.youtube.com/watch?v=5SaUE9lQ1Pc&t=1s)
