# ROS - pygradientpolytope

This repository includes nodes for generating and visualizing Cartesian Available : (1) Velocity Polytope (2) Force Polytope (3) Desired Polytope
(4) Performance Index - Capacity Margin in [Rviz](http://wiki.ros.org/rviz).

## Installation
This repository was tested on KUKA KR4 R600 Sixx robot with [KUKA RSI](https://docs.quanser.com/quarc/documentation/kuka_rsi_block.html) package installed in it. The robot is controllled through an external PC with [ROS Noetic](http://wiki.ros.org/noetic) and Ubuntu 20.04 LTS. 
### Hardware requirements
* External PC. Our specifications were;
```
ntel® Core™ i7-10700K CPU @ 3.80GHz × 16
RAM: 16 GB
Graphics: NVIDIA Corporation GP106GL [Quadro P2200]
```
* KUKA Robot. We used KUKA KR6 R700 Sixx.
* Gripper. 

### Software and Library Requirements 

* Ubuntu 20.04 LTS
* ROS Noetic
If you are new to ROS, go [here](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) to learn how to create a catkin workspace. 
* KUKA RSI package installed in the KUKA robot.
* Polytope. This is the source repository for polytope, a toolbox for geometric operations on polytopes in any dimension.
[Polytope](https://pypi.org/project/polytope/).
* [pykdl_utils](http://wiki.ros.org/pykdl_utils) Higher Level Python wrapper for PyKDL in ROS for Kinematic Solver
* Python native libraries [Scipy](https://scipy.org/), [Numpy](https://numpy.org/)
* Polygon ROS geometry messages for plotting in Rviz [jsk-ros-pkg](https://github.com/jsk-ros-pkg/jsk_recognition)
* [KUKA experimental](https://gitlab.com/imr-robotics/kuka_experimental) Digital output interface is included in this repository. Official repository [here](https://github.com/ros-industrial/kuka_experimental). Instructions for KR C4 robot controller [here](https://github.com/ros-industrial/kuka_experimental/tree/indigo-devel/kuka_rsi_hw_interface/krl/KR_C4). 
* Polytope ros message publisher forked and modified from Pycapacity Library [capacity_visual_utils](https://github.com/askuric/polytope_vertex_search/blob/master/ROS_nodes/panda_capacity/scripts/capacity/capacity_visual_utils.py)
In a Terminal
```
$ cd ~/catkin_ws/src/
$ git clone https://gitlab.com/KeerthiSagarSN/rospygradientpolytope

