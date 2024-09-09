import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import mplib
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

from dex_retargeting import yourdfpy as urdf


import mplib
from mplib import Pose




def load_robot():
    pass


def start_retargeting(queue: multiprocessing.Queue, pc_queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    
    logger.info(f"Start retargeting with config {config_path}")
    
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    
    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

    # Camera
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    # urdf_path = "/home/rancho/2-ldl/Robot_py/dex-retargeting/pi_code/a1arm_description/urdf/a1arm.urdf"
    
    urdf_path = "/home/rancho/2-ldl/Robot_py/dex-retargeting/pi_code/a1arm_description/a1arm_mplib.urdf"
    
    filepath = Path(urdf_path)
    robot_name = filepath.stem
    loader.scale = 1
    
    filepath = str(filepath)
    robot = loader.load(filepath)

    robot.set_root_pose(sapien.Pose([0, 0, 0.0], [1, 0, 0, 0]))


    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    print(sapien_joint_names) # ['arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6', 'gripper1_axis', 'gripper2_axis']
    
    
    robot_model = RobotWrapper(filepath)
    
    init_qpos = robot.get_qpos()
    
    # filepath = "/".join(filepath.split("/")[:-2])
    # print(filepath)
    srdf_path = "/home/rancho/2-ldl/Robot_py/dex-retargeting/pi_code/a1arm_description/a1arm_mplib_mplib.srdf"
    planner = mplib.Planner(
            urdf=filepath,
            srdf=srdf_path,
            move_group="arm_segee",
        )
    
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.02, 0.02, 0.06])
    builder.add_box_visual(half_size=[0.02, 0.02, 0.06], material=[1, 0, 0])
    red_cube = builder.build(name="red_cube")
    red_cube.set_pose(sapien.Pose([0.4, 0.3, 0.06]))

    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.02, 0.02, 0.04])
    builder.add_box_visual(half_size=[0.02, 0.02, 0.04], material=[0, 1, 0])
    green_cube = builder.build(name="green_cube")
    green_cube.set_pose(sapien.Pose([0.2, -0.3, 0.04]))

    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.02, 0.02, 0.07])
    builder.add_box_visual(half_size=[0.02, 0.02, 0.07], material=[0, 0, 1])
    blue_cube = builder.build(name="blue_cube")
    blue_cube.set_pose(sapien.Pose([0.6, 0.1, 0.07]))
    
    poses = [
            Pose([0.4, 0.3, 0.12], [0, 1, 0, 0]),
            Pose([0.2, -0.3, 0.08], [0, 1, 0, 0]),
            Pose([0.6, 0.1, 0.14], [0, 1, 0, 0]),
        ]
    
    def move_to_pose(pose):
        result = planner.plan_screw(
            pose,
            robot.get_qpos(),
            time_step=1 / 250,
        )
        if result["status"] == "Success":
            follow_path(result)
        else:
            move_to_pose_with_RRTConnect(pose)
            
    def move_to_pose_with_RRTConnect(pose: Pose):
        
        print("plan_pose")
        result = planner.plan_pose(pose, robot.get_qpos(), time_step=1 / 250)
        # plan_pose ankor end
        if result["status"] != "Success":
            print(result["status"])
            return -1
        # do nothing if the planning fails; follow the path if the planning succeeds
        follow_path(result)
        return 0        
            
    def follow_path(result):
        n_step = result["position"].shape[0]
        if n_step == 0:
            return
        end = result["position"][-1]
        print(f"{end}")
        
        arm_qpose = robot.get_qpos()
        for i in range(len(end)):
            arm_qpose[i] = end[i]
        
        robot.set_qpos(arm_qpose)
        
        viewer.render()
        viewer.render()
        # time.sleep(10)
        return
        
        active_joints = robot.get_active_joints()
        print(f"follow_path, n_step:{n_step}")
        for i in range(n_step):
            # print(f"{i}")
            # qf = robot.compute_passive_force(
            #     gravity=True, coriolis_and_centrifugal=True
            # )
            # robot.set_qf(qf)
            # set the joint positions and velocities for move group joints only.
            # The others are not the responsibility of the planner
            for j in range(len(planner.move_group_joint_indices)):
                active_joints[j].set_drive_target(result["position"][i][j])
                active_joints[j].set_drive_velocity_target(
                    result["velocity"][i][j]
                )
            # simulation step
            scene.step()
            # render every 4 simulation steps to make it faster
            if i % 20 == 0:
                scene.update_render()
                viewer.render()
                
            arm_qpose = robot.get_qpos()
            # print(f"arm_qpose:{arm_qpose}")
        
        
    # qpose = Pose(p=[0,0,0.5])
    # move_to_pose(qpose)
    
    griper_pose = [0, 0.03]
    
    while True:
        arm_pose = robot.get_pose()
        arm_qpose = robot.get_qpos()
        print(f"arm_pose:{arm_pose}") # arm_pose:Pose([-2.32831e-10, 9.54969e-12, 3.72529e-09], [1, 0, 0, 0])
        print(f"arm_qpose:{arm_qpose}") # arm_qpose:[0. 0. 0. 0. 0. 0. 0. 0.]
        # raise
        
        
        arm_qpose[6] += 0.01
        # robot.set_qpos(arm_qpose)

        for i in range(3):
            pose = poses[i]
            # pose.p[2] += 0.2
            pose = Pose(p=[0,0,1.0])
            # move_to_pose(pose,)
            
            
            result = planner.IK(pose, robot.get_qpos(),)
            print(result)
            if result[0] == 'Success':
                goal_qpos = result[1][0]
                # goal_qpos[-1] = 0.03
                # goal_qpos[-2] = 0.03
                robot.set_qpos(goal_qpos)
                
            for joint in robot.active_joints[-2:]:
                # print(joint.get_pose_in_child())
                joint.set_drive_target(0)
            for i in range(100):
                qf = robot.compute_passive_force(
                    gravity=True, coriolis_and_centrifugal=True
                )
                robot.set_qf(qf)
                scene.step()
                if i % 4 == 0:
                    scene.update_render()
                    viewer.render()
            
            # self.open_gripper()
            # pose.p[2] -= 0.12
            # self.move_to_pose(pose)
            # self.close_gripper()
            # pose.p[2] += 0.12
            # self.move_to_pose(pose)
            # pose.p[0] += 0.1
            # self.move_to_pose(pose)
            # pose.p[2] -= 0.12
            # self.move_to_pose(pose)
            # self.open_gripper()
            # pose.p[2] += 0.12
            # self.move_to_pose(pose)



        for _ in range(2):
            viewer.render()


def produce_frame(queue: multiprocessing.Queue, pc_queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    cam = DepthCam()
    
    

    while True:
        if not cam.get_frames():
            continue
            
            
        bgr_image = cv2.cvtColor(cam.color_image, cv2.COLOR_BGRA2BGR)
        time.sleep(1 / 30.0)
        cam.get_point_cloud()
        
        pc_queue.put(cam.point_cloud)
        queue.put(bgr_image)
        

# python demo_control.py --robot-name panda  --retargeting-type dexpilot  --hand-type left
# python demo_control.py --robot-name shadow  --retargeting-type dexpilot  --hand-type left
# vector/dexpilot 
def main(
    # robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType, camera_path: Optional[str] = None
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    # config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    # robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    # robot_dir = Path(__file__).absolute().parent.parent / "assets" / "robots" / "hands"

    config_path = ""
    robot_dir = Path(__file__).absolute().parent / "a1arm_description"


    queue = multiprocessing.Queue(maxsize=1)
    pc_queue = multiprocessing.Queue(maxsize=1)
    # producer_process = multiprocessing.Process(target=produce_frame, args=(queue, pc_queue, camera_path))
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(queue, pc_queue, str(robot_dir), str(config_path)))

    # producer_process.start()
    consumer_process.start()

    # producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
