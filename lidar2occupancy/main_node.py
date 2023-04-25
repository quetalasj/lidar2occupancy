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


class Lidar2Occupancy(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')

        # make manual approximate parallel to floor transformation
        self.transformation = np.eye(4)
        self.transformation[2, -1] = 1
        self.transformation[:3, :3] = R.from_euler('x', 25, degrees=True).as_matrix()
        self._is_first_frame = True
        self.subscription = self.create_subscription(PointCloud2, '/points', self.listener_callback,
                                                     qos_profile_system_default)
        self.publisher = self.create_publisher(PointCloud2, '/processed_points', qos_profile_system_default)
        self.publisher_floor = self.create_publisher(PointCloud2, '/floor', qos_profile_system_default)

    def listener_callback(self, msg):
        lidar_data = read_points(msg)
        np_pts = np.array([lidar_data['x'], lidar_data['y'], lidar_data['z']])
        if self._is_first_frame:
            self._is_first_frame = False
            self.refine_transformation(np_pts)

        aligned_pts = np_pts.T @ self.transformation[:3, :3] + self.transformation[:3, -1]
        _, floor_np, no_floor_np = self.find_floor(aligned_pts, 0.1)

        processed_msg = create_cloud_xyz32(
            Header(stamp=self.get_clock().now().to_msg(), frame_id=msg.header.frame_id),
            # np.asarray(pcd.points)
            aligned_pts
        )

        floor_msg = create_cloud_xyz32(
            Header(stamp=self.get_clock().now().to_msg(), frame_id=msg.header.frame_id),
            floor_np
        )

        # new_msg.data = msg.data
        # msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(processed_msg)
        self.publisher_floor.publish(floor_msg)

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
        """ Refines initial coarse transformation estimating angle between Z axis and floor normal"""
        approximately_transformed_pts = msg_np_pts.T @ self.transformation[:3, :3] + self.transformation[:3, -1]
        # approximately_floor = approximately_transformed_pts[np.abs(approximately_transformed_pts[:, -1] < 1)]
        # pcd = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(approximately_floor)
        # )
        # plane_model, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=10000)
        # floor_pcd = pcd.select_by_index(inliers)
        plane_model, _, _ = self.find_floor(approximately_transformed_pts)
        plane_normal = plane_model[:-1]
        y_rot = self.vectors_angle(plane_normal[[0, -1]])
        x_rot = self.vectors_angle(plane_normal[[1, -1]])
        x_R = R.from_euler('x', x_rot, degrees=False).as_matrix()
        y_R = R.from_euler('y', y_rot, degrees=False).as_matrix()
        self.transformation[:3, :3] = self.transformation[:3, :3] @ x_R @ y_R
        # return floor_pcd.points


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = Lidar2Occupancy()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
