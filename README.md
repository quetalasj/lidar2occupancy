#### Problem description
A robot during its motion scans the area. To prevent collision it stops in front of an obstacle. 
Unfortunately, there are situations when floor points are considered as obstacles (false positive obstacle detection), and robot stops. 
So there is a fundamental need to filter these points out of the point cloud.

`ros2 run lidar2occupancy lidar2occupancy`

Only the front points should be used to create a lidar scan from the point cloud:

The points are projected to the unit circle, then points corresponding to the same angle are chosen (points that are approximately on similiary directed lidar rays).
Only the closest one is added to the lidar scan among the chosen batch of points.

#### Result
![alt text](result.png)

#### Performed steps
1. Floor/ground detection, segmentation on point cloud
2. Scan orientation parallel to the ground
3. Floor points filtering
4. Robot Region of Interest cropping
5. Points discretization, projection to plane
6. Point Cloud transformation to Laser Scan
7. Occupancy Grid generation
