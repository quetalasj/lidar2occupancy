import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from rclpy.qos import qos_profile_system_default
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs_py.point_cloud2 import read_points, create_cloud_xyz32
import open3d as o3d
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Lidar2Occupancy(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')

        # make manual approximate parallel to floor transformation
        self.transformation = np.eye(4)
        self.transformation[2, -1] = 1
        self.transformation[:3, :3] = R.from_euler('x', 25, degrees=True).as_matrix()
        self._initial_transform = np.copy(self.transformation)
        self._is_first_frame = True
        self.subscription = self.create_subscription(PointCloud2, '/points', self.listener_callback,
                                                     qos_profile_system_default)
        self._coarse_transform_pub = self.create_publisher(PointCloud2, '/coarse_transform_pts', qos_profile_system_default)
        self._fine_transform_pub = self.create_publisher(PointCloud2, '/fine_transform_pts', qos_profile_system_default)
        self._floor_pub = self.create_publisher(PointCloud2, '/floor', qos_profile_system_default)
        self._no_floor_pub = self.create_publisher(PointCloud2, '/no_floor', qos_profile_system_default)
        self._processed_pts_pub = self.create_publisher(PointCloud2, '/processed_pts', qos_profile_system_default)
        self._projected_pts_pub = self.create_publisher(PointCloud2, '/projected_pts', qos_profile_system_default)
        self._projected_pts_pub_ring = self.create_publisher(PointCloud2, '/projected_pts_ring', qos_profile_system_default)
        self._image_pub = self.create_publisher(Image, "/image", qos_profile_system_default)
        # image frame
        self.cv_bridge = CvBridge()
        self._image = np.zeros(np.round((4 / 0.05, 4 / 0.05)).astype(int)).astype(np.uint8)

    def listener_callback(self, msg):
        lidar_data = read_points(msg)
        np_pts = np.array([lidar_data['x'], lidar_data['y'], lidar_data['z']])
        if self._is_first_frame:
            self._is_first_frame = False
            self.refine_transformation(np_pts)

        aligned_pts = np_pts.T @ self.transformation[:3, :3] + self.transformation[:3, -1]
        _, floor_np, no_floor_np = self.find_floor(aligned_pts, 0.1)

        processed_pts_np = no_floor_np[(no_floor_np[:, -1] > 0.) *
                                       (np.abs(no_floor_np[:, -1]) < 2.) *
                                       (np.abs(no_floor_np[:, 0]) < 2.) *
                                       (np.abs(no_floor_np[:, 1]) < 2.)]
        # remove some robot points
        processed_pts_np = processed_pts_np[np.linalg.norm(processed_pts_np[:, :-1], axis=1) > 0.9]
        processed_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(processed_pts_np)
        )
        # clean up the points from outliers
        processed_pcd, _ = processed_pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=1)
        # project to ground plane
        projected_pts_np = np.copy(np.asarray(processed_pcd.points))
        projected_pts_np[:, -1] = 0
        projected_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(projected_pts_np)
        )

        projected_pcd = projected_pcd.voxel_down_sample(0.05)
        discretized_pts_np = np.copy(np.asarray(projected_pcd.points))
        discretized_pts_np[:, -1] = 0
        # generate image
        image = np.copy(self._image)

        pixels = np.round(
            ((2, 2) + discretized_pts_np[:, :-1]) * (1/4 * 79, 1/4 * 79) # normalize to image plane
        ).astype(int)
        pixels_num = pixels[:, 0] * 80 + pixels[:, 1]
        image.reshape(-1)[pixels_num] = 255
        image.reshape((80, 80))

        # visualize all results
        # coarse_transform_msg = create_cloud_xyz32(
        #     Header(stamp=self.get_clock().now().to_msg(), frame_id=msg.header.frame_id),
        #     np_pts.T @ self._initial_transform[:3, :3] + self._initial_transform[:3, -1]
        # )
        # fine_transform_msg = create_cloud_xyz32(
        #     Header(stamp=self.get_clock().now().to_msg(), frame_id=msg.header.frame_id),
        #     aligned_pts
        # )
        # floor_msg = create_cloud_xyz32(
        #     Header(stamp=self.get_clock().now().to_msg(), frame_id=msg.header.frame_id),
        #     floor_np
        # )
        # no_floor_msg = create_cloud_xyz32(
        #     Header(stamp=self.get_clock().now().to_msg(), frame_id=msg.header.frame_id),
        #     no_floor_np
        # )
        processed_pts_msg = create_cloud_xyz32(
            Header(stamp=self.get_clock().now().to_msg(), frame_id=msg.header.frame_id),
            np.asarray(processed_pcd.points)
        )
        projected_pts_msg = create_cloud_xyz32(
            Header(stamp=self.get_clock().now().to_msg(), frame_id=msg.header.frame_id),
            discretized_pts_np
        )

        projected_pts_msg_ring = create_cloud_xyz32(
            Header(stamp=self.get_clock().now().to_msg(), frame_id=msg.header.frame_id),
            discretized_pts_np / np.linalg.norm(discretized_pts_np, axis=1)[..., None]
        )
        # import pdb
        # pdb.set_trace()
        # new_msg.data = msg.data
        # msg.header.stamp = self.get_clock().now().to_msg()
        # self._coarse_transform_pub.publish(coarse_transform_msg)
        # self._fine_transform_pub.publish(fine_transform_msg)
        # self._floor_pub.publish(floor_msg)
        # self._no_floor_pub.publish(no_floor_msg)
        self._processed_pts_pub.publish(processed_pts_msg)
        self._projected_pts_pub.publish(projected_pts_msg)
        self._projected_pts_pub_ring.publish(projected_pts_msg_ring)
        # import pdb
        # pdb.set_trace()
        self._image_pub.publish(self.cv_bridge.cv2_to_imgmsg(image, "mono8"))

    def vectors_angle(self, a, b=np.array([0, 1])):
        """ Estimates angles between 2d vectors"""
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        return np.arccos(np.clip(np.dot(a_norm, b_norm), -1, 1))

    def find_floor(self, lidar_scan_np, z_threshold=1.):
        mask = np.abs(lidar_scan_np[:, -1] < z_threshold)
        approximately_floor = lidar_scan_np[mask]
        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(approximately_floor)
        )
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=10000)
        floor_np = approximately_floor[inliers]
        no_floor_mask = np.ones(lidar_scan_np.shape[0], dtype=bool)
        # no_floor_mask[mask][inliers] = False
        np.put_along_axis(no_floor_mask[mask], np.array(inliers), False, axis=0)
        all_indices = np.arange(lidar_scan_np.shape[0])
        mask_indices = np.copy(all_indices[mask])
        mask_indices[inliers] = -1
        all_indices[mask] = mask_indices

        return plane_model, floor_np, lidar_scan_np[all_indices]

    def refine_transformation(self, msg_np_pts):
        # also can be done with o3d.geometry.PointCloud.orient_normals_to_align_with_direction
        """ Refines initial coarse transformation estimating angle between Z axis and floor normal"""
        approximately_transformed_pts = msg_np_pts.T @ self.transformation[:3, :3] + self.transformation[:3, -1]
        plane_model, floor_np, _ = self.find_floor(approximately_transformed_pts)
        plane_normal = plane_model[:-1]
        y_rot = self.vectors_angle(plane_normal[[0, -1]])
        x_rot = self.vectors_angle(plane_normal[[1, -1]])
        x_R = R.from_euler('x', x_rot, degrees=False).as_matrix()
        y_R = R.from_euler('y', y_rot, degrees=False).as_matrix()
        self.transformation[:3, :3] = self.transformation[:3, :3] @ x_R @ y_R
        z_shift = np.mean(floor_np @ x_R @ y_R, axis=0)[-1]
        self.transformation[2, -1] -= z_shift
        # return floor_pcd.points


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = Lidar2Occupancy()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
