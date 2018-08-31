#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    point_cloud = ros_to_pcl(pcl_msg)
    
    """Statistical Outlier Filtering"""

    # The threshold scale factor
    x = 0.40
    neighboring_points = 10
 
    outlier_filter = point_cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(neighboring_points)
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()

    # TODO: Voxel Grid Downsampling
    LEAF_SIZE = 0.01

    vox = cloud_filtered.make_voxel_grid_filter()
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    cloud_filtered = vox.filter()

    """Passthrough of the y axis"""
    filter_axis = 'y'
    axis_min = -0.40
    axis_max = 0.40

    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)

    cloud_filtered = passthrough.filter()

    """Passthrough of the z axis"""
    filter_axis = 'z'
    axis_min = 0.6
    axis_max = 1.1

    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)

    cloud_filtered = passthrough.filter()

    # TODO: RANSAC Plane Segmentation
    max_distance = 0.01

    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(max_distance)

    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    extracted_table = cloud_filtered.extract(inliers, negative=False)
    extracted_objects = cloud_filtered.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_objects) # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()

    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(10000)
    ec.set_SearchMethod(tree)

    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(extracted_objects) 
    ros_cloud_table = pcl_to_ros(extracted_table)
    ros_cloud_cluster = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cloud_cluster)

# Exercise-3 TODOs:

    nr_of_bins = 64

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = extracted_objects.extract(pts_list)

        # Compute the associated feature vector
        ros_cloud   = pcl_to_ros(pcl_cluster)
        chists      = compute_color_histograms(ros_cloud, nr_of_bins, using_hsv=True)
        normals     = get_normals(ros_cloud)
        nhists      = compute_normal_histograms(normals, nr_of_bins)
        feature     = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cloud
        detected_objects.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    detected_objects_pub.publish(detected_objects)

    try:
        pr2_mover(detected_objects)
        return
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    output_list = []

    # Get the set object list
    object_list_param = rospy.get_param('/object_list')

    # Create the Labels and Centroids
    centroids = {}

    # Aggregate the centroids of the recognized objects
    for object in object_list:

        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids[object.label] = np.mean(points_arr, axis=0)[:3]

    # Quick info about which objects we expect and which we got.
    rospy.loginfo(
        'Expected {} and got: {}'.format(
            [object_params['name'] for object_params in object_list_param], 
            [object.label for object in object_list]
        )
    )

    for object_params in object_list_param:

        name = object_params['name']

        # Abort the iteration if no recognized object for this listed object has been found.
        if name not in centroids.keys():
            continue

        object_group = object_params['group']

        object_name = String()
        object_name.data = name

        # Convert numpy float64 to python floats
        centroid = [np.asscalar(x) for x in centroids[name]] 

        # Store the object position
        pose = Pose()
        pose.position.x = centroid[0]
        pose.position.y = centroid[1]
        pose.position.z = centroid[2]

        # Decide which arm to use based on the object group
        arm_to_use = String()
        arm_to_use.data = 'right' if object_group == 'green' else 'left'

        # Set manually according to the test world
        scene_num = Int32()
        scene_num.data = 3

        obj_yaml_dict = make_yaml_dict(
            scene_num, 
            arm_to_use,
            object_name, 
            pose,
            Pose()
        )
        output_list.append(obj_yaml_dict)

    send_to_yaml('output_%i.yaml' % scene_num.data, output_list)



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('perception_routine', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
