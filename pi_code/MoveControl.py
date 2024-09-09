import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import mplib.kinematics
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.robot_wrapper import RobotWrapper
from HandDetection import MultiHandDetector
from Depthcam import DepthCam

import rospy
import tf.transformations as tftr

import mplib
from mplib import Pose



import tf2_ros
from geometry_msgs.msg import TransformStamped




class MoveControl:
    def __init__(self) -> None:
        # self._setup_scene()
        
        pass

    def setup_scene(self):
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
        
        # Setup
        self.scene = sapien.Scene()
        render_mat = sapien.render.RenderMaterial()
        render_mat.base_color = [0.06, 0.08, 0.12, 1]
        render_mat.metallic = 0.0
        render_mat.roughness = 0.9
        render_mat.specular = 0.8
        self.scene.add_ground(0, render_material=render_mat, render_half_size=[1000, 1000])

        # Lighting
        self.scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
        self.scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
        self.scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
        self.scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
        self.scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

        # Camera
        cam = self.scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
        cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

        self.viewer = Viewer()
        self.viewer.set_scene(self.scene)
        self.viewer.control_window.show_origin_frame = False
        self.viewer.control_window.move_speed = 0.01
        self.viewer.control_window.toggle_camera_lines(False)
        self.viewer.set_camera_pose(cam.get_local_pose())


    def load_robot(self, ):
        if not hasattr(self, 'scene'):
            raise ("call 'self.setup_scene()' first")
        loader = self.scene.create_urdf_loader()
        self.urdf_path = "/home/rancho/2-ldl/Robot_py/dex-retargeting/pi_code/a1arm_description/a1arm_mplib.urdf"
    
        filepath = Path(self.urdf_path)
        robot_name = filepath.stem
        loader.scale = 1
        self.robot = loader.load(self.urdf_path)
        sapien_joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        print(sapien_joint_names) # ['arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6', 'gripper1_axis', 'gripper2_axis']
        robot_model = RobotWrapper(self.urdf_path)
        self.init_qpos = self.robot.get_qpos()
        self.srdf_path = "/home/rancho/2-ldl/Robot_py/dex-retargeting/pi_code/a1arm_description/a1arm_mplib_mplib.srdf"
    
        
        
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        # self.links = self.robot.get_links()
        self.active_joints = self.robot.get_active_joints()
        self.ee_name = "arm_joint6"
        self.griper_base_joint = self.robot.find_joint_by_name(self.ee_name)
        self.ee_index = self.active_joints.index(self.griper_base_joint) + 1
        
        for joint in self.active_joints:
            # joint.set_drive_property(
            #     stiffness=1000,
            #     damping=200,
            # )
            joint.set_drive_property(stiffness=1000, damping=0, force_limit=100000, mode="force")
        # links = self.robot.get_links()
        # for link in links:
        #     link.
            
    def setup_planner(self):
        self.planner = mplib.Planner(
                urdf=self.urdf_path,
                srdf=self.srdf_path,
                move_group="arm_seg6",
            )
        
    def init_ros_pub(self, ):
        self.ros_broadcaster = tf2_ros.TransformBroadcaster()
        
        # 定义坐标系变换信息
        static_transformStamped = TransformStamped()
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
        quat = tftr.quaternion_from_euler(roll, yaw, pitch)

        # 将四元数设置到 static_transformStamped
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]

        # 发布坐标系变换
        self.ros_broadcaster.sendTransform(static_transformStamped)
    
    def build_objects(self,):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.06])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.06], material=[1, 0, 0])
        red_cube = builder.build(name="red_cube")
        red_cube.set_pose(sapien.Pose([0.4, 0.3, 0.06]))

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.04])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.04], material=[0, 1, 0])
        green_cube = builder.build(name="green_cube")
        green_cube.set_pose(sapien.Pose([0.2, -0.3, 0.04]))

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.07])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.07], material=[0, 0, 1])
        blue_cube = builder.build(name="blue_cube")
        blue_cube.set_pose(sapien.Pose([0.6, 0.1, 0.07]))
        
        self.cubes = [red_cube, green_cube, blue_cube]
        
        self.obj_poses = [
                Pose([0.4, 0.3, 0.12], [0, 1, 0, 0]),
                Pose([0.2, -0.3, 0.08], [0, 1, 0, 0]),
                Pose([0.6, 0.1, 0.14], [0, 1, 0, 0]),
            ]
    
    def set_gripper(self, pos: float):
        """
        Helper function to activate gripper joints
        Args:
            pos: position of the gripper joint in real number
        """
        # The following two lines are particular to the panda robot
        direct = False
        if direct:
            # goal_qpos[-1] = 0.03
            # goal_qpos[-2] = 0.03
            # self.robot.set_qpos(goal_qpos)
            target_qpos = self.robot.get_qpos()
            target_qpos[-1] = pos
            target_qpos[-2] = pos
            self.robot.set_qpos(target_qpos)
        else:
            for joint in self.active_joints[-2:]:
                # print(joint.get_name())
                joint.set_drive_target(pos)
                # print(joint.drive_target)
            qpos = self.robot.get_qpos()
            
            for joint in self.active_joints[:-2]:
                index = self.active_joints.index(joint)
                joint.set_drive_target(qpos[index])
                
            # 100 steps is plenty to reach the target position
            for i in range(100):
                qf = self.robot.compute_passive_force(
                    gravity=True, coriolis_and_centrifugal=True
                )
                self.robot.set_qf(qf)
                self.scene.step()
                if i % 4 == 0:
                    self.scene.update_render()
                    self.viewer.render()
    
    def open_gripper(self):
        self.set_gripper(0.03)

    def close_gripper(self):
        self.set_gripper(0.00)
    
    
    def move_to_pose(self, pose, with_screw=True):
        """API to multiplex between the two planning methods"""
        if with_screw:
            return self.move_to_pose_with_screw(pose)
        else:
            return self.move_to_pose_with_RRTConnect(pose)
            
    def move_to_pose_with_screw(self, pose):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            pose,
            self.robot.get_qpos(),
            time_step=1 / 250,
        )
        if result["status"] == "Success":
            self.follow_path(result)
            return 0
        else:
            # fall back to RRTConnect if the screw motion fails (say contains collision)
            return self.move_to_pose_with_RRTConnect(pose)
            
    def move_to_pose_with_RRTConnect(self, pose: Pose):
        
        print("plan_pose")
        result = self.planner.plan_pose(pose, self.robot.get_qpos(), time_step=1 / 250)
        # plan_pose ankor end
        if result["status"] != "Success":
            print(result["status"])
            return -1
        # do nothing if the planning fails; follow the path if the planning succeeds
        self.follow_path(result)
        return 0        
            
    def follow_path(self, result):
        n_step = result["position"].shape[0]
        if n_step == 0:
            return
        active_joints = self.robot.get_active_joints()
        print(f"follow_path, n_step:{n_step}")
        for i in range(n_step):
            for j in range(len(self.planner.move_group_joint_indices)):
                active_joints[j].set_drive_target(result["position"][i][j])
                active_joints[j].set_drive_velocity_target(
                    result["velocity"][i][j]
                )
            # simulation step
            self.scene.step()
            # render every 4 simulation steps to make it faster
            if i % 4 == 0:
                # global_pose = self.griper_base_joint.get_global_pose()
                # pub_ros_tf(global_pose, "gripper_base")
                # link_pose = self.planner.pinocchio_model.get_link_pose(6)
                # pub_ros_tf(link_pose, "link_pose")
                
                ee_global_pose = self.get_ee_joint_world_pose()
                link_pose_cal = demo.transfer_ee_jointpose_to_linkpose(ee_global_pose)
                link_pose_true = demo.get_ee_link_world_pose()
                demo.pub_ros_tf(ee_global_pose, "gripper_base")
                demo.pub_ros_tf(link_pose_cal, "link_pose")
                demo.pub_ros_tf(link_pose_true, "link_pose_target")
                
                self.scene.update_render()
                self.viewer.render()
                
            arm_qpose = self.robot.get_qpos()
            # print(f"arm_qpose:{arm_qpose}")
        print("follow_path end")
    
    def stop_timestamps(self, timestamps):
        for i in range(timestamps):
            self.scene.step()
            # render every 4 simulation steps to make it faster
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()
        
        
    # qpose = Pose(p=[0,0,0.5])
    # move_to_pose(qpose)
    
    def step(self):
        arm_pose = self.robot.get_pose()
        arm_qpose = self.robot.get_qpos()
        print(f"arm_pose:{arm_pose}") # arm_pose:Pose([-2.32831e-10, 9.54969e-12, 3.72529e-09], [1, 0, 0, 0])
        print(f"arm_qpose:{arm_qpose}") # arm_qpose:[0. 0. 0. 0. 0. 0. 0. 0.]
        
        target_pose = Pose(p=[0,0,1.0])
            
        result = self.planner.IK(target_pose, self.robot.get_qpos(),)
            
        if result[0] == 'Success':
            goal_qpos = result[1][0]
            # goal_qpos[-1] = 0.03
            # goal_qpos[-2] = 0.03
            self.robot.set_qpos(goal_qpos)
            
        # for joint in robot.active_joints[-2:]:
        #     # print(joint.get_pose_in_child())
        #     joint.set_drive_target(0)
        # for i in range(100):
        #     qf = robot.compute_passive_force(
        #         gravity=True, coriolis_and_centrifugal=True
        #     )
        #     robot.set_qf(qf)
        #     scene.step()
        #     if i % 4 == 0:
        #         scene.update_render()
        #         viewer.render()
            
        self.scene.update_render()
        self.viewer.render()

    def quaternion_conjugate(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
    def transfer_ee_jointpose_to_linkpose(self, global_pose:sapien.pysapien.Pose):
        q_y_90 = np.array([-np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0])
        q_new = self.quaternion_multiply(global_pose.q, q_y_90, )
        link_pose = global_pose
        link_pose.q = q_new
        return link_pose
        
        
    def pub_ros_tf(self, global_pose:sapien.pysapien.Pose, name):
        ts = TransformStamped()
        ts.header.frame_id = "world"
        ts.header.stamp = rospy.Time.now()

        ts.child_frame_id = name

        ts.transform.translation.x = global_pose.p[0]
        ts.transform.translation.y = global_pose.p[1]
        ts.transform.translation.z = global_pose.p[2]

        # r, p, y = global_pose.rpy
        # print(f"RPY:{global_pose.rpy}")
        # qtn = tftr.quaternion_from_euler(r, p, y)
        # print(f"qtn:{qtn}")
        
        
        # ts.transform.rotation.x = qtn[0]
        # ts.transform.rotation.y = qtn[1]
        # ts.transform.rotation.z = qtn[2]
        # ts.transform.rotation.w = qtn[3]
        
        ts.transform.rotation.x = global_pose.q[1]
        ts.transform.rotation.y = global_pose.q[2]
        ts.transform.rotation.z = global_pose.q[3]
        ts.transform.rotation.w = global_pose.q[0]
        
        # 发布数据
        self.ros_broadcaster.sendTransform(ts)

    def get_ee_joint_world_pose(self):
        ee_global_pose = self.griper_base_joint.get_global_pose()
        return ee_global_pose
    
    def cal_ee_link_world_pose(self):
        ee_global_pose = self.griper_base_joint.get_global_pose()
        link_pose_cal = self.transfer_ee_jointpose_to_linkpose(ee_global_pose)
        return link_pose_cal
    
    def get_ee_link_world_pose(self):
        # print(self.ee_index)
        ee_link_pose = demo.planner.pinocchio_model.get_link_pose(self.ee_index)
        return ee_link_pose

def grab_cubes():
    pass




if __name__ == "__main__":
    rospy.init_node('coordinate_frame_broadcaster')
    demo = MoveControl()
    demo.setup_scene()
    demo.load_robot()
    demo.setup_planner()
    demo.build_objects()
    demo.init_ros_pub()



    # 动态坐标系
    roll, pitch, yaw = 0, 0, 0,
    euler = roll, pitch, yaw
    # euler = np.radians([45, 30, 60])
    quaternion = tftr.quaternion_from_euler(*euler)
    x,y,z,w = quaternion
    q=[w,x,y,z]
    pose1 = Pose(p=[0.2,0.1,0.8])
    # pose1 = Pose([0.0127772, 0.00826558, 0.124648], [-0.025529, 0.99961, 0.00173005, -0.0111596])
    
    
    # res = demo.planner.IK(pose1, demo.robot.get_qpos(),n_init_qpos=100)
    # print(res)
    
    for joint in demo.active_joints:
        print(f"joint {joint.name}, type: {joint.type}, target={joint.get_drive_target()}")
        print(f"{joint.get_global_pose()}---{joint.get_limits()}")
        
        
    
    for link in demo.robot.get_links():
        print(f"link {link.get_name(), link.get_entity_pose()}")


    
    
    poses = []
    poses.append(Pose([-0.0277244, -0.0607818, 0.427498], [-0.103336, -0.027808, 0.826011, -0.553403]))
    poses.append(Pose([0.0807502, -0.0408548, 0.362079], [0.435247, 0.253993, 0.86158, 0.0610619]))
    # poses.append(Pose([-0.0236857, -0.0883946, 0.55868], [-0.175863, 0.884052, 0.394414, 0.178778]))
    poses.append(Pose([0.15461, -0.283679, 0.157487], [-0.0461306, -0.229109, -0.971111, -0.0482041]))
    count = 0
    
    manual_control = True
    while True and not rospy.is_shutdown():
        if manual_control:
            
            ee_global_pose = demo.get_ee_joint_world_pose()
            print(f"G:{ee_global_pose}")
            link_pose_cal = demo.transfer_ee_jointpose_to_linkpose(ee_global_pose)
            print(f"L cal :{link_pose_cal}")
            
            demo.move_to_pose(link_pose_cal, ) 
            
            link_pose_true = demo.get_ee_link_world_pose()
            print(f"L true:{link_pose_true}")
            
            demo.pub_ros_tf(ee_global_pose, "gripper_base")
            demo.pub_ros_tf(link_pose_true, "link_pose")
            
            # demo.stop_timestamps(1)
            demo.scene.update_render()
            demo.viewer.render()
            
        else:
            target_pose = poses[count%len(poses)]
            count += 1
            
            ee_global_pose = demo.get_ee_joint_world_pose()
            print(f"G:{ee_global_pose}")
            link_pose_cal = demo.transfer_ee_jointpose_to_linkpose(ee_global_pose)
            print(f"L cal :{link_pose_cal}")
            link_pose_true = demo.get_ee_link_world_pose()
            print(f"L true:{link_pose_true}")
            
            demo.pub_ros_tf(ee_global_pose, "gripper_base")
            demo.pub_ros_tf(link_pose_true, "link_pose")
            
            
            demo.stop_timestamps(100)
            demo.move_to_pose(target_pose, ) 
            # print(demo.planner.pinocchio_model.get_link_names())
            # for link_id, _ in enumerate(demo.planner.pinocchio_model.get_link_names()):
            #     print(f"Link pose:{demo.planner.pinocchio_model.get_link_pose(link_id)}")
            
            
            demo.open_gripper()
            demo.stop_timestamps(300)
            # demo.move_to_pose(pose2, )
            # print(f"quaternion: {pose2.q}")
            # demo.close_gripper()
            # demo.stop_timestamps(300)