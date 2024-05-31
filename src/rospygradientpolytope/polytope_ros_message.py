### Polytope ROS message generation made easy with Antun Skuric's library for Polytope
'''
https://github.com/askuric/polytope_vertex_search/blob/master/ROS_nodes/panda_capacity/scripts/capacity/capacity_visual_utils.py

Revision made to the original file 

@author: keerthi
keerthi.sagar@imr.ie
'''
import rospy
# time evaluation
import time
from sensor_msgs.msg import JointState,PointCloud
from geometry_msgs.msg import Polygon, Point32, PolygonStamped, PointStamped
from jsk_recognition_msgs.msg import PolygonArray,SegmentArray,Segment
from std_msgs.msg import Header,Float64
from visualization_msgs.msg import Marker
import tf
import numpy as np

def create_vertex_msg(force_vertex, pose, frame, scaling_factor = 500):
    pointcloud_message = PointCloud()
    for i in range(force_vertex.shape[1]):
        point = Point32()
        point.x = force_vertex[0,i]/scaling_factor + pose[0]
        point.y = force_vertex[1,i]/scaling_factor + pose[1]
        point.z = force_vertex[2,i]/scaling_factor + pose[2]
        pointcloud_message.points.append(point)
    
    # polytop stamped message
    pointcloud_message.header = Header()
    pointcloud_message.header.frame_id = frame
    pointcloud_message.header.stamp = rospy.Time.now()
    return pointcloud_message



def create_capacity_vertex_msg(capacity_vertex, pose, frame, scaling_factor):
    pointstamped_message = PointStamped()
    point = Point32()
    #for i in range(len(capacity_vertex)):        
    point.x = capacity_vertex[0]/scaling_factor + pose[0]
    point.y = capacity_vertex[1]/scaling_factor + pose[1]
    point.z = capacity_vertex[2]/scaling_factor + pose[2]
    pointstamped_message.point = point
        
    # polytop stamped message
    pointstamped_message.header = Header()
    pointstamped_message.header.frame_id = frame
    pointstamped_message.header.stamp = rospy.Time.now()
    return pointstamped_message


def create_polytopes_with_vertex_msg(polytope_verts, pose, frame, scaling_factor):
    polygonarray_message = PolygonArray()
    polygonarray_message.header = Header()
    polygonarray_message.header.frame_id = frame
    polygonarray_message.header.stamp = rospy.Time.now()
    ## Need to initialize all points at once instead of append
    ## Lots of overhead for int conversion too !!! TODO
    ## Append may be causing latency - TODO
    vert_counter = 0
    for kk in range(int(np.shape(polytope_verts)[0]/3.0)):
        polygon_message = Polygon()        
        for i in range(3):
            point = Point32()            

            point.x = (polytope_verts[vert_counter,0]/(scaling_factor*1.0)) + pose[0]            
            point.y = (polytope_verts[vert_counter,1]/(scaling_factor*1.0)) + pose[1]
            point.z = (polytope_verts[vert_counter,2]/(scaling_factor*1.0)) + pose[2]
            
            polygon_message.points.append(point)
            vert_counter+=1
        
        # polytope stamped message
        polygon_stamped = PolygonStamped()
        polygon_stamped.polygon = polygon_message
        polygon_stamped.header = Header()
        polygon_stamped.header.frame_id = frame
        polygon_stamped.header.stamp = rospy.Time.now()
        polygonarray_message.polygons.append(polygon_stamped)
        polygonarray_message.likelihood.append(1.0)
        
    return polygonarray_message

def create_polytopes_msg(polytope_verts,polytope_faces, pose, frame, scaling_factor):
    polygonarray_message = PolygonArray()
    polygonarray_message.header = Header()
    polygonarray_message.header.frame_id = frame
    polygonarray_message.header.stamp = rospy.Time.now()
    ## Need to initialize all points at once instead of append
    ## Lots of overhead for int conversion too !!! TODO
    ## Append may be causing latency - TODO
    for face_polygon in polytope_faces:
        polygon_message = Polygon()        
        for i in range(len(face_polygon)):
            point = Point32()            

            point.x = (polytope_verts[int(face_polygon[i]),0]/(scaling_factor*1.0)) + pose[0]            
            point.y = (polytope_verts[int(face_polygon[i]),1]/(scaling_factor*1.0)) + pose[1]
            point.z = (polytope_verts[int(face_polygon[i]),2]/(scaling_factor*1.0)) + pose[2]
            
            polygon_message.points.append(point)
        
        # polytope stamped message
        polygon_stamped = PolygonStamped()
        polygon_stamped.polygon = polygon_message
        polygon_stamped.header = Header()
        polygon_stamped.header.frame_id = frame
        polygon_stamped.header.stamp = rospy.Time.now()
        polygonarray_message.polygons.append(polygon_stamped)
        polygonarray_message.likelihood.append(1.0)
    return polygonarray_message
def create_fast_polytopes_msg(polytope_verts,polytope_faces, pose, frame, scaling_factor):
    polygonarray_message = PolygonArray()
    polygonarray_message.header = Header()
    polygonarray_message.header.frame_id = frame
    polygonarray_message.header.stamp = rospy.Time.now()
    ## Need to initialize all points at once instead of append
    ## Lots of overhead for int conversion too !!! TODO
    ## Append may be causing latency - TODO
    time_start = rospy.Time.now()
    for face_polygon in polytope_faces:
        polygon_message = Polygon()        
        for i in range(len(face_polygon)):
            point = Point32()            
            face_polygon_number = int(face_polygon[i]) ## It is an int
            point.x = (polytope_verts[face_polygon_number,0]/(scaling_factor*1.0)) + pose[0]            
            point.y = (polytope_verts[face_polygon_number,1]/(scaling_factor*1.0)) + pose[1]
            point.z = (polytope_verts[face_polygon_number,2]/(scaling_factor*1.0)) + pose[2]
            
            polygon_message.points.append(point)
        
        # polytope stamped message
        polygon_stamped = PolygonStamped()
        polygon_stamped.polygon = polygon_message
        polygon_stamped.header = Header()
        polygon_stamped.header.frame_id = frame
        polygon_stamped.header.stamp = rospy.Time.now()
        polygonarray_message.polygons.append(polygon_stamped)
        polygonarray_message.likelihood.append(1.0)
    time_end = rospy.Time.now()
    polytope_message_duration = time_end - time_start

    print('polytope_message_duration',polytope_message_duration)
    return polygonarray_message

## Only one face of the polygon is here
def create_polygon_msg(polytope_verts,polytope_faces, pose, frame, scaling_factor):
    polygonarray_message = PolygonArray()
    polygonarray_message.header = Header()
    polygonarray_message.header.frame_id = frame
    polygonarray_message.header.stamp = rospy.Time.now()
    ## Need to initialize all points at once instead of append
    ## Append may be causing latency - TODO
    polygon_message = Polygon()

    for face_polygon in polytope_faces:

        point = Point32()            

        point.x = (polytope_verts[int(face_polygon),0]/(scaling_factor*1.0)) + pose[0]            
        point.y = (polytope_verts[int(face_polygon),1]/(scaling_factor*1.0)) + pose[1]
        point.z = (polytope_verts[int(face_polygon),2]/(scaling_factor*1.0)) + pose[2]
        
        polygon_message.points.append(point)
        
        # polytope stamped message
        polygon_stamped = PolygonStamped()
        polygon_stamped.polygon = polygon_message
        polygon_stamped.header = Header()
        polygon_stamped.header.frame_id = frame
        polygon_stamped.header.stamp = rospy.Time.now()
        polygonarray_message.polygons.append(polygon_stamped)
        polygonarray_message.likelihood.append(1.0)
    return polygonarray_message

## Draw line segment between two point stamped ROS - message
def create_segment_msg(point_1, point_2,pose, frame, scaling_factor):

    ## Initialize Line segment message
    linestampedarray_message = SegmentArray()
    


    start_pt = Point32()
    start_pt.x = point_1[0]/scaling_factor + pose[0]       
    start_pt.y = point_1[1]/scaling_factor + pose[1]
    start_pt.z = point_1[2]/scaling_factor + pose[2]


    end_pt = Point32()
    end_pt.x = point_2[0]/scaling_factor + pose[0]       
    end_pt.y = point_2[1]/scaling_factor + pose[1]
    end_pt.z = point_2[2]/scaling_factor + pose[2]

    line_segment = Segment()
    line_segment.start_point = start_pt
    line_segment.end_point = end_pt
    linestampedarray_message.segments = [line_segment]



    linestampedarray_message.header = Header()
    linestampedarray_message.header.frame_id = frame
    linestampedarray_message.header.stamp = rospy.Time.now()

    return linestampedarray_message

def create_ellipsoid_msg(S, U, pose, frame, scaling_factor = 500):
    # calculate rotation matrix
    Rot_f = np.identity(4)
    Rot_f[0:3,0:3] = U

    marker = Marker()
    marker.header.frame_id = frame
    marker.pose.position.x = pose[0]
    marker.pose.position.y = pose[1]
    marker.pose.position.z = pose[2]
    quaternion = tf.transformations.quaternion_from_matrix( Rot_f )
    #type(pose) = geometry_msgs.msg.Pose
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]

    marker.type = marker.SPHERE
    marker.color.g = 0.7
    marker.color.r = 1.0
    marker.color.a = 0.5
    marker.scale.x = 2*S[0]/scaling_factor
    marker.scale.y = 2*S[1]/scaling_factor
    marker.scale.z = 2*S[2]/scaling_factor
    return marker




if __name__ == '__main__':
    polytope_ros_message()