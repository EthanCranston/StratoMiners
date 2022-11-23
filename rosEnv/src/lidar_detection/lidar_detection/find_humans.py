from re import A
import numpy as np
import numpy.lib.recfunctions as rfn 
from sklearn import cluster

import os
import yaml
from yaml.loader import SafeLoader

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
# from lidar_detection_msgs.msg import PointCloudWithConidence, PointCloudWithConidenceArray

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# Begin from ros2_numpy

import sys
import array

Image.encoding

# prefix to the names of dummy fields we add to get byte alignment
# correct. this needs to not clash with any actual field names
DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)


def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(
                ('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_to_nptype[f.datatype].itemsize * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list


def dtype_to_fields(dtype):
    '''Convert a numpy record datatype into a list of PointFields.
    '''
    fields = []
    for field_name in dtype.names:
        np_field_type, field_offset = dtype.fields[field_name]
        pf = PointField()
        pf.name = field_name
        if np_field_type.subdtype:
            item_dtype, shape = np_field_type.subdtype
            pf.count = int(np.prod(shape))
            np_field_type = item_dtype
        else:
            pf.count = 1

        pf.datatype = nptype_to_pftype[np_field_type]
        pf.offset = field_offset
        fields.append(pf)
    return fields


def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray
    Reshapes the returned array to have shape (height, width), even if the
    height is 1.
    The reason for using np.frombuffer rather than struct.unpack is
    speed... especially for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (
            fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


def array_to_pointcloud2(cloud_arr, stamp=None, frame_id=None):
    '''Converts a numpy record array to a sensor_msgs.msg.PointCloud2.
    '''
    # make it 2d (even if height will be 1)
    cloud_arr = np.atleast_2d(cloud_arr)

    cloud_msg = PointCloud2()

    if stamp is not None:
        cloud_msg.header.stamp = stamp
    if frame_id is not None:
        cloud_msg.header.frame_id = frame_id
    cloud_msg.height = cloud_arr.shape[0]
    cloud_msg.width = cloud_arr.shape[1]
    cloud_msg.fields = dtype_to_fields(cloud_arr.dtype)
    cloud_msg.is_bigendian = sys.byteorder != 'little'
    cloud_msg.point_step = cloud_arr.dtype.itemsize
    cloud_msg.row_step = cloud_msg.point_step*cloud_arr.shape[1]
    cloud_msg.is_dense = \
        all([np.isfinite(
            cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])

    # The PointCloud2.data setter will create an array.array object for you if you don't
    # provide it one directly. This causes very slow performance because it iterates
    # over each byte in python.
    # Here we create an array.array object using a memoryview, limiting copying and
    # increasing performance.
    memory_view = memoryview(cloud_arr)
    if memory_view.nbytes > 0:
        array_bytes = memory_view.cast("B")
    else:
        # Casting raises a TypeError if the array has no elements
        array_bytes = b""
    as_array = array.array("B")
    as_array.frombytes(array_bytes)
    cloud_msg.data = as_array
    return cloud_msg


def merge_rgb_fields(cloud_arr):
    '''Takes an array with named np.uint8 fields 'r', 'g', and 'b', and returns
       an array in which they have been merged into a single np.float32 'rgb'
       field. The first byte of this field is the 'r' uint8, the second is the
       'g', uint8, and the third is the 'b' uint8.
       This is the way that pcl likes to handle RGB colors for some reason.
    '''
    r = np.asarray(cloud_arr['r'], dtype=np.uint32)
    g = np.asarray(cloud_arr['g'], dtype=np.uint32)
    b = np.asarray(cloud_arr['b'], dtype=np.uint32)
    rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)

    # not sure if there is a better way to do this. i'm changing the type of
    # the array from uint32 to float32, but i don't want any conversion to take
    # place -jdb
    rgb_arr.dtype = np.float32

    # create a new array, without r, g, and b, but with rgb float32 field
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if field_name not in ('r', 'g', 'b'):
            new_dtype.append((field_name, field_type))
    new_dtype.append(('rgb', np.float32))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'rgb':
            new_cloud_arr[field_name] = rgb_arr
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]

    return new_cloud_arr


def split_rgb_field(cloud_arr):
    '''Takes an array with a named 'rgb' float32 field, and returns an array in
    which this has been split into 3 uint 8 fields: 'r', 'g', and 'b'.
    (pcl stores rgb in packed 32 bit floats)
    '''
    rgb_arr = cloud_arr['rgb'].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb_arr & 255, dtype=np.uint8)

    # create a new array, without rgb, but with r, g, and b fields
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if not field_name == 'rgb':
            new_dtype.append((field_name, field_type))
    new_dtype.append(('r', np.uint8))
    new_dtype.append(('g', np.uint8))
    new_dtype.append(('b', np.uint8))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'r':
            new_cloud_arr[field_name] = r
        elif field_name == 'g':
            new_cloud_arr[field_name] = g
        elif field_name == 'b':
            new_cloud_arr[field_name] = b
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]
    return new_cloud_arr


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
    a 3xN matrix.
    '''
    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & \
            np.isfinite(cloud_array['y']) & \
            np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']

    return points


def pointcloud2_to_xyz_array(cloud_msg, remove_nans=True):
    return get_xyz_points(
        pointcloud2_to_array(cloud_msg), remove_nans=remove_nans)

# End from ros2_numpy


class LidarDetection(Node):

    def __init__(self):
        super().__init__("LidarDetection")

        # Load YAML file
        cfg = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'src/lidar_detection/config/lidar_detection.yaml')
        with open(cfg) as f:
            data = yaml.load(f, Loader=SafeLoader)
            point_cloud_topic_ = data['lidar_detection']['point_cloud_topic']
            human_points_topic_ = data['lidar_detection']['human_points_topic']
            clouds_with_confidence_ = data['lidar_detection']['clouds_with_confidence']
            self.transform_frame_ = data['lidar_detection']['transform_frame']
            self.working_frame_ = data['lidar_detection']['working_frame']
            self.human_template_ = data['lidar_detection']['human_template']
            self.confidence_min_ = float(data['lidar_detection']['confidence_minimum'])

        # Load template
        self.templateResolution = 10
        # self.get_logger().info(f'DIR: {os.getcwd()}')
        self.template = np.load(f'src/lidar_detection/lidar_detection/templates/{self.human_template_}')

        # Define subscriber
        self.subscription = self.create_subscription(
            PointCloud2,
            point_cloud_topic_,
            self.lidar_callback,
            10
        )

        # Define publisher
        self.pointsPublisher = self.create_publisher(
            PointCloud2,
            human_points_topic_,
            10
        )

        # Define publisher
        # self.conidencePublisher = self.create_publisher(
        #     PointCloudWithConidenceArray,
        #     clouds_with_confidence_,
        #     10
        # )


        # Define TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def lidar_callback(self, cloud: PointCloud2):
        # Define point cloud in self
        self.point_cloud_ = cloud

        # Locate humans
        new_cloud = pointcloud2_to_xyz_array(cloud)
        humanPoints, clusterToConfidence = self.find_human_points(new_cloud)
        if humanPoints is None: 
            self.get_logger().info("LIDAR located no humans")
            return
        cloudsWithConfidences = []
        for cluster, confidence in clusterToConfidence.items():
            if confidence >= self.confidence_min_:
                self.get_logger().info(f'LIDAR confidence for cluster {cluster} is {confidence}')
                # xyzArray = rfn.unstructured_to_structured(humanPoints[np.where(humanPoints[:, 4] == cluster), :3], np.dtype([(n, np.float32) for n in ['x', 'y', 'z']]))
                # sinlgeHumanCloud = array_to_pointcloud2(xyzArray, frame_id=self.transform_frame_, stamp=cloud.header.stamp)

                # PointCloudWithConidence.clu
                

        xyzArray = rfn.unstructured_to_structured(humanPoints[:, :3], np.dtype([(n, np.float32) for n in ['x', 'y', 'z']]))

        humanCloud = array_to_pointcloud2(xyzArray, frame_id=self.transform_frame_, stamp=cloud.header.stamp)
        self.pointsPublisher.publish(humanCloud)


        self.get_logger().info(f"Located {len(np.unique(humanPoints[:, 3]))} human clusters")

    def get_pointclouds(self, array: np.ndarray) -> list[PointCloud2]:
        clouds = []
        for cluster in range(np.max(humanPoints[:, 3])):
            clusterPoints = array[array[:, 3] == cluster]
            clouds.append(array_to_pointcloud2(clusterPoints))
        
        return clouds

    def create_template(self, array):
        # normalize
        for col in range(3):
            array[:, col] = (array[:, col]-np.min(array[:, col]))/(np.max(array[:, col])-np.min(array[:, col]))

        array = array * (self.templateResolution-2)

        comp = np.zeros([self.templateResolution, self.templateResolution, self.templateResolution])

        for row in array:
            X = int(np.floor(row[0]))
            Y = int(np.floor(row[1]))
            Z = int(np.floor(row[2]))
            for xOff in range(2):
                for yOff in range(2):
                    for zOff in range(2):
                        comp[X+xOff, Y+yOff, Z+zOff] += np.sqrt((row[0]-X+xOff-1)**2 + (row[1]-Y+yOff-1)**2 + (row[2]-Y+yOff-1)**2)

        # normalize
        comp = (comp-np.min(comp))/(np.max(comp)-np.min(comp))

        return comp

    def get_template_difference(self, array):
        comparisonTemplate = self.create_template(array)
        diffArray = np.reshape(np.abs(self.template - comparisonTemplate), -1)
        diff = sum(diffArray[np.where(diffArray > 0.75)])
        maxDiff = 1
        for s in self.template.shape: maxDiff *= s
        
        return diff / maxDiff

    def clean_data(self, array: np.ndarray) -> np.ndarray:
        '''
            Removes outlier points
            Removed floor
        '''
        distance = np.sqrt(array[:, 0]**2 + array[:, 1]**2 + array[:, 2]**2)
        avgDist = np.average(distance)
        stdDist = np.std(distance)
        zScores = (distance - avgDist) / stdDist
        array = array[~(abs(zScores) > 1)]

        try:
            # Using TF frames, transform the base
            base_transform = self.tf_buffer.lookup_transform(
                self.transform_frame_,
                self.working_frame_,
                self.point_cloud_.header.stamp)

            # Using the transformed base, find the floor
            floor = base_transform.transform.translation.z

            # Using found floor, cut off the floor
            array = array[np.logical_and(array[:, 2], array[:, 2] > floor)]
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.transform_frame_} to {self.working_frame_}: {ex}')

        return array


    def get_human_confidence(self, points: np.ndarray) -> float:
        dist = [0, 0, 0]
        std = [0, 0, 0]

        for i in range(3):
            dist[i] = max(points[:,i]) - min(points[:,i])
            std[i] = np.std(points[:,i])
        
        if not (0.5 < dist[2] < 2): # height check
            return 0
        if (max(dist[:2]) > 1): # check width
            return 0
        if (max(dist[:2]) < 0.3): # check width
            return 0
        if (max(std[:2]) > 0.2): # horizontal standard deviation
            return 0

        threshold = 0.005
        diff = self.get_template_difference(points)

        if  diff > threshold:
            return 0

        confidence = (threshold - diff) / threshold
        return confidence


    # def find_humans(self, lidarPoints: np.ndarray, **kwargs) -> np.ndarray:
    def find_human_points(self, lidarPoints: np.ndarray) -> np.ndarray:
        '''
            Input:
            lidarPoints: npArray -> lidar points in form [[x, y, z], ...]

            Output:
            npArray -> human points and which cluster they are a part of in form [[x, y, z, cluster num], ...]
        '''
        # Clean the data
        array = self.clean_data(lidarPoints)

        # Cluster humans using DBSCAN
        dbscan = cluster.DBSCAN(eps=0.3, algorithm="kd_tree", n_jobs=-1)
        dbscan.fit(array)
        y_pred = dbscan.labels_.astype(int)


        humanClusters = []
        clusterToConfidence = {}
        humanPoints = np.zeros(array.shape[0])
        for clusterI in range(max(y_pred)+1):
            points = array[y_pred==clusterI]
            clusterToConfidence[clusterI] = self.get_human_confidence(points)
            if  clusterToConfidence[clusterI] >= self.confidence_min_:
                humanClusters.append(clusterI)
                humanPoints = np.logical_or(humanPoints, y_pred==clusterI)
        if len(humanClusters) > 0:
            array = array[humanPoints]
            y_pred = y_pred[humanPoints]
            humanPointsWithCluster = np.insert(array, 3, y_pred, axis=1)
        else:
            humanPointsWithCluster = None

        

        return humanPointsWithCluster, clusterToConfidence



def main(args=None):
    rclpy.init(args=args)

    lidar_detection = LidarDetection()
    try: 
        rclpy.spin(lidar_detection)
        rclpy.shutdown()
    except KeyboardInterrupt:
        exit()


if __name__ == '__main__':
    main()
