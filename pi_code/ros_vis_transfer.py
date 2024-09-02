import rospy
import numpy as np
import tf.transformations


import tf2_ros
import geometry_msgs.msg



class Trans(object):
    def __init__(self, camera_position, camera_orientation) -> None:
        self._init_tf_transfer(camera_position, camera_orientation)
        
    def _init_tf_transfer(self, camera_position, camera_orientation):
        
        self.camera_position = np.array(camera_position)  # 相机在地面坐标系中的位置
        self.camera_orientation = np.array(camera_orientation)  # 相机在地面坐标系中的欧拉角
        
        croll, cpitch, cyaw = self.camera_orientation
        # 构建变换矩阵
        t_camera = self.camera_position
        self.R_camera = tf.transformations.euler_matrix(croll, cpitch, cyaw)[:3, :3]
        T_camera_to_ground = np.eye(4)
        T_camera_to_ground[:3, :3] = self.R_camera
        T_camera_to_ground[:3, 3] = t_camera
        self.T_camera_to_ground = T_camera_to_ground 
        
    def transfer_point(self, point):
        return np.dot(self.T_camera_to_ground, np.concatenate((point, [1]), axis=0))[:3]
    
    def transfer_pose(self, p_camera):
        # p_camera = np.array([x, y, z, roll, pitch, yaw])
        # 1. 计算相机坐标系到地面基坐标系的变换矩阵
        self.T_camera_to_ground

        # 2. 将物体从相机坐标系转换到地面基坐标系
        p_ground_position = np.dot(self.T_camera_to_ground, np.concatenate((p_camera[:3], [1])))[:3]

        # 3. 将物体姿态从相机坐标系转换到地面基坐标系
        R_object = tf.transformations.euler_matrix(p_camera[3], p_camera[4], p_camera[5])[:3, :3]
        R_object_ground = np.dot(self.R_camera, R_object)
        p_ground_orientation = tf.transformations.euler_from_matrix(R_object_ground)

        # 4. 合并位置和姿态信息
        p_ground = np.concatenate((p_ground_position, p_ground_orientation))
        
        # print(p_ground)
        return p_ground
    
    
    
    

def main():
    rospy.init_node('coordinate_frame_broadcaster')

    # 创建 StaticTransformBroadcaster
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    # 定义坐标系变换信息
    static_transformStamped = geometry_msgs.msg.TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "base_link"
    static_transformStamped.child_frame_id = "camera_link"

    # 设置位置和姿态
    static_transformStamped.transform.translation.x = 1
    static_transformStamped.transform.translation.y = 0
    static_transformStamped.transform.translation.z = 1
    
    roll = np.pi/2
    yaw = 0
    pitch = np.pi/2
    quat = tf.transformations.quaternion_from_euler(roll, yaw, pitch)

    # 将四元数设置到 static_transformStamped
    static_transformStamped.transform.rotation.x = quat[0]
    static_transformStamped.transform.rotation.y = quat[1]
    static_transformStamped.transform.rotation.z = quat[2]
    static_transformStamped.transform.rotation.w = quat[3]

    # 发布坐标系变换
    broadcaster.sendTransform(static_transformStamped)
    
    
    
    # 定义 camera_link 到 hand_link 的坐标系变换
    static_transformStamped2 = geometry_msgs.msg.TransformStamped()
    static_transformStamped2.header.stamp = rospy.Time.now()
    static_transformStamped2.header.frame_id = "camera_link"
    static_transformStamped2.child_frame_id = "hand_link"

    # 设置 camera_link 到 hand_link 的位置和姿态
    static_transformStamped2.transform.translation.x = 0.0
    static_transformStamped2.transform.translation.y = 0.0
    static_transformStamped2.transform.translation.z = 1
    roll2 = 0.1
    yaw2 = 0.2
    pitch2 = 0.3
    quat2 = tf.transformations.quaternion_from_euler(roll2, yaw2, pitch2)
    static_transformStamped2.transform.rotation.x = quat2[0]
    static_transformStamped2.transform.rotation.y = quat2[1]
    static_transformStamped2.transform.rotation.z = quat2[2]
    static_transformStamped2.transform.rotation.w = quat2[3]
    
    
    
    broadcaster.sendTransform(static_transformStamped2)
    
    
    

    rospy.spin()

if __name__ == '__main__':
    main()