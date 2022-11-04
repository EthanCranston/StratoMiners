from re import A
import numpy as np
import cv2

import os
import yaml
from yaml.loader import SafeLoader

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge


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


class LidarCV(Node):

    def __init__(self):
        super().__init__("LidarCV")

        # Load YAML file
        cfg = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'src/lidar_cv/config/lidar_cv.yaml')
        with open(cfg) as f:
            data = yaml.load(f, Loader=SafeLoader)
            point_cloud_topic_ = data['lidar_cv']['point_cloud_topic']
            found_humans_topic_ = data['lidar_cv']['found_humans_topic']

        # Define subscriber
        self.subscription = self.create_subscription(
            PointCloud2,
            point_cloud_topic_,
            self.lidar_callback,
            10
        )

        # Define image publisher
        self.img_publisher = self.create_publisher(
            Image,
            found_humans_topic_,
            10
        )

    def lidar_callback(self, cloud: PointCloud2):
        new_cloud = pointcloud2_to_xyz_array(cloud)
        found_humans = self.find_humans(new_cloud)

        for (x, y, z, conf) in found_humans:
            self.get_logger().info(
                f"Found human at ({x}, {y}, {z}) with confidence {conf}")

    def cartesian_to_polar(self, x, y):
        r = np.sqrt(x**2 + y**2)
        t = np.arctan2(x, y)

        return r, t

    def polar_to_cartesian(self, r, t):
        x = r * np.cos(t)
        y = r * np.sin(t)

        return x, y

    def get_weighted_difference(self, points):
        diffs = [0 for x in points]

        for i, point in enumerate(points):
            for i2, otherPoint in enumerate(points):
                if i == i2:
                    continue
                diffs[i] += abs(point - otherPoint) / (i - i2)**(1.5)

        return diffs

    def annotated_lidar_img(self, baseImg, annotationInfo, show):
        resizeX = 10
        resizeY = 4

        displayImg = baseImg[:, :, [0, 0, 0]]
        displayImg = cv2.resize(displayImg, None, fx=resizeX, fy=resizeY)

        for obj in annotationInfo:
            xPoint = (obj[0][0] * resizeX, obj[0][1] * resizeY)
            yPoint = (obj[1][0] * resizeX, obj[1][1] * resizeY)
            confidence = "{:.2f}".format(obj[2])

            displayImg = cv2.rectangle(
                displayImg, xPoint, yPoint, (0, 0, 255), 2)
            cv2.putText(displayImg, confidence, xPoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1)

        # Using cv_bridge, encode image
        # https://github.com/eric-wieser/ros_numpy/issues/13
        bridge = CvBridge()
        rosImg = bridge.cv2_to_imgmsg(displayImg.round().astype(np.uint8))

        # Publish and actively refresh image
        cv2.waitKey(1)
        self.img_publisher.publish(rosImg)

        # Show annotated lidar image as a cv window if prompted
        if show:
            cv2.imshow("lidar", displayImg)

    def interpolate(self, array: np.ndarray, ranges: list[tuple]) -> tuple[np.array, list[int], list[int]]:
        # Interpolation is performed manually instead of using `np.interp()` so that the scale and offset can be saved
        assert len(array[0]) == len(ranges)

        scaleFactors = [None, None, None]
        offsets = [None, None, None]

        for i in range(len(ranges)):
            scaleFactors[i] = (ranges[i][1] - ranges[i][0]) / \
                (array[:, i].max() - array[:, i].min())
            offsets[i] = ranges[i][0] - array[:, i].min()
            array[:, i] = array[:, i] * scaleFactors[i] + \
                (offsets[i] * scaleFactors[i])

        return array, scaleFactors, offsets

    def unInterpolate(self, array: np.ndarray, scaleFactors, offsets):
        assert len(array[0]) == len(scaleFactors)

        for i in range(len(scaleFactors)):
            array[:, i] = (array[:, i] - offsets[i] *
                           scaleFactors[i]) / scaleFactors[i]

        return array

    def generate_binary_slice(self, image, sliceStart, sliceEnd):
        # thresholds, then open & closes the image to isolate a areas of nearby points
        _, low_img = cv2.threshold(image, sliceStart, 1, cv2.THRESH_BINARY)
        _, high_img = cv2.threshold(image, sliceEnd, 1, cv2.THRESH_BINARY_INV)

        binnedImage = cv2.bitwise_and(low_img, high_img)

        binnedImage = cv2.morphologyEx(
            binnedImage, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 30)))
        binnedImage = cv2.morphologyEx(
            binnedImage, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 20)))

        return binnedImage

    def find_matching_components(self, image, sliceDepth, targetSize, threshold):
        annotationInfo = []
        centroidCoords = []
        for start in np.arange(0, 1, sliceDepth):
            binaryImg = self.generate_binary_slice(
                image, start, start + sliceDepth)
            _, _, values, centroids = cv2.connectedComponentsWithStats(
                np.uint8(binaryImg))

            targetArea = min(targetSize / np.arctan(start + 0.00001), 1000)

            ySize = binaryImg.shape[1]
            for i in range(len(values)):
                area = values[i][4]
                if abs(area - targetArea) < threshold:
                    top, left, height, width = [values[i][x] for x in range(4)]
                    confidence = 1 - abs(area - targetArea) / threshold
                    centroidCoords.append(
                        [ySize - centroids[i][1], centroids[i][0], start + sliceDepth/2])
                    annotationInfo.append(
                        [(int(top), int(left)), (int(top + height), int(left + width)), confidence])

        centroidCoords = np.array(centroidCoords)
        annotationInfo = np.array(annotationInfo)

        return centroidCoords, annotationInfo

    def find_humans(self, lidarPoints: np.ndarray, **kwargs) -> np.ndarray:
        '''
        Input:
        lidarPoints: npArray -> lidar points in form [[x, y, z], ...]

        Kwargs:
        size: int -> relative area of a human
        sizeThreshold: int -> The acceptable variation from the expected size
        sliceDepth: float on interval [0,1] -> percentage of depth that is used in each slice
        showImg: bool -> toggles showing an annotated image for debugging


        Output:
        npArray -> Center point of detected humans and confidence interval in the form [[x, y, z, confidence], ...]

        Notes:
        x & y = floor plane (The direction of x and y are relative to sensor position)
        z = vertical
        r = radius/depth from sensor 
        t = theta - angle with respect to senor
        '''
        humanSize = kwargs.get("size", 50)
        sizeThreshold = kwargs.get("thresh", 100)
        sliceDepth = kwargs.get("sliceDepth", 0.1)
        showImg = kwargs.get("showImg", False)

        # [x, y, z] -> [z, y, x]
        imageArray = lidarPoints[:, [2, 1, 0]]

        # This order is used because it is what displays nicely as an image
        # [z, y, x] -> [z, t, r]
        imageArray[:, 2], imageArray[:, 1] = self.cartesian_to_polar(
            imageArray[:, 2], imageArray[:, 1])

        # remove far back points #TODO
        imageArray = np.delete(imageArray, np.where(
            imageArray[:, 2] > 15), axis=0)

        imageSize = [100, 100]  # TODO

        # Interpolates coordinates to image size and color depth
        imageArray, scaleFactors, offsets = self.interpolate(
            imageArray, [(0, imageSize[0] - 1), (0, imageSize[1] - 1), (0, 1)])

        bwImage = np.zeros((imageSize[0], imageSize[1], 1))
        # Assigns array to image values #TODO
        for row in imageArray:
            bwImage[imageSize[0] - int(row[0]) - 1, int(row[1])] = [row[2]]

        centroidCoords, annotationInfo = self.find_matching_components(
            bwImage, sliceDepth, humanSize, sizeThreshold)

        # Scale points back to original scene
        centerPoints = self.unInterpolate(
            centroidCoords, scaleFactors, offsets)
        # [z, t, r] -> [z, y, x]
        centerPoints[:, 2], centerPoints[:, 1] = self.polar_to_cartesian(
            centerPoints[:, 2], centerPoints[:, 1])
        # [z, y, x] -> [x, y, z]
        centerPoints = centerPoints[:, [2, 1, 0]]

        # Adds confidence intervals
        pointsAndConfidence = np.insert(
            centerPoints, 3, annotationInfo[:, 2], axis=1)

        # Publish/show annotated lidar image
        self.annotated_lidar_img(bwImage, annotationInfo, showImg)
        return pointsAndConfidence


def main(args=None):
    rclpy.init(args=args)

    lidar_cv = LidarCV()

    rclpy.spin(lidar_cv)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
