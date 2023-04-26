#### Problem description
When Robot drives around, the lidar scans the area and gets points. 
To prevent collision robot stops in front of the obstacle (points in a point cloud). 
Unfortunately, there are situations when points from floor are considered an obstacle, and robots stop. 
So there is a fundamental need to filter these points out of the point cloud.

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
