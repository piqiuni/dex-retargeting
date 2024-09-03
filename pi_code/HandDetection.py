import math
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
from example.vector_retargeting.single_hand_detector import *
import cv2
from loguru import logger

import tf.transformations


class Hand(object):
    def __init__(self) -> None:
        self.bool = False
        self.heading = 0
        self.index_tip_wc = []
        self.thumd_tip_wc = []
        self.wrist_point_ic = []
        self.wrist_point_wc = []
        self.wirst_cam_coord = np.array([0.0,0.0,0.0])
        self.wirst_world_coord = np.array([0.0,0.0,0.0])
        self.wirst_pose_angle = np.array([0.0,0.0,0.0])
        self.wirst_pose = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self.transform_matrix = np.eye(4, dtype=np.float32)
        
        self.grasp_distance = 0
        
        self.name_dict()
        self.num_box = 0
        self.joint_pos = None
        self.keypoint_2d = None
        self.mediapipe_wrist_rot = None
        
    def name_dict(self):
        self.namedict = {
                            "0": "WRIST",
                            "1": "THUMB_CMC",
                            "2": "THUMB_MCP",
                            "3": "THUMB_IP",
                            "4": "THUMB_TIP",
                            "5": "INDEX_FINGER_MCP",
                            "6": "INDEX_FINGER_PIP",
                            "7": "INDEX_FINGER_DIP",
                            "8": "INDEX_FINGER_TIP",
                            "9": "MIDDLE_FINGER_MCP",
                            "10": "MIDDLE_FINGER_PIP",
                            "11": "MIDDLE_FINGER_DIP",
                            "12": "MIDDLE_FINGER_TIP",
                            "13": "RING_FINGER_MCP",
                            "14": "RING_FINGER_PIP",
                            "15": "RING_FINGER_DIP",
                            "16": "RING_FINGER_TIP",
                            "17": "PINKY_MCP",
                            "18": "PINKY_PIP",
                            "19": "PINKY_DIP",
                            "20": "PINKY_TIP"
                        }


    def update_none(self):
        self.bool = False
        self.num_box = 0
        self.joint_pos = None
        self.keypoint_2d = None
        self.mediapipe_wrist_rot = None

    def update_hand(self, result, index, joints):
        self.bool = True
        num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = joints
        self.num_box = num_box
        self.joint_pos = joint_pos
        self.keypoint_2d = keypoint_2d
        self.mediapipe_wrist_rot = mediapipe_wrist_rot
        
        index_tip = result.multi_hand_world_landmarks[index].landmark[8]
        self.index_tip_wc = [index_tip.x, index_tip.y, index_tip.z]
        
        thumb_tip = result.multi_hand_world_landmarks[index].landmark[4]
        self.thumb_tip_wc = [thumb_tip.x, thumb_tip.y, thumb_tip.z]
        self.grasp_distance = self.get_distance(self.index_tip_wc, self.thumb_tip_wc)
        
        # print(f"self.index_tip: {self.index_tip_wc}")
        # print(f"self.thumb_tip: {self.thumb_tip_wc}")
        
        wrist_ic = result.multi_hand_landmarks[index].landmark[0]
        self.wrist_point_ic = [wrist_ic.x, wrist_ic.y, wrist_ic.z]
        wrist_wc = result.multi_hand_world_landmarks[index].landmark[0]
        self.wrist_point_wc = [wrist_wc.x, wrist_wc.y, wrist_wc.z]
        # print(f"self.wrist_point_wc: {self.wrist_point_wc}")
        # print(f"self.wrist_point_ic: {self.wrist_point_ic}")
        
        # print(self.mediapipe_wrist_rot)
        R = np.array(self.mediapipe_wrist_rot)
        
        roll, pitch, yaw = tf.transformations.euler_from_matrix(R)
        self.wrist_angle= roll, pitch, yaw
        

    def get_distance(self, pos1, pos2):
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        
        return distance

    def get_hand_world_pose(self, ):
        pass


class MultiHandDetector(SingleHandDetector):
    def __init__(self, min_detection_confidence=0.8, min_tracking_confidence=0.8, selfie=False):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hand_list = ['Left', 'Right']
        
        
        self.selfie = selfie
        self.operator2mano_left = OPERATOR2MANO_LEFT
        self.operator2mano_right = OPERATOR2MANO_RIGHT
        
        self.inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        # self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]
    
        self.left_hand = Hand()
        self.right_hand = Hand()
    
        self.MARGIN = 10  # pixels
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
        
        self._init_tf_transfer()
        self._init_hand_transfer()
    
    def _init_tf_transfer(self, ):
        
        self.camera_position = np.array([0, 0, 0])  # 相机在地面坐标系中的位置
        self.camera_orientation = np.array([np.pi/2, 0, -np.pi/2])  # 相机在地面坐标系中的欧拉角
        
        croll, cpitch, cyaw = self.camera_orientation
        # 构建变换矩阵
        t_camera = self.camera_position
        self.R_camera = tf.transformations.euler_matrix(croll, cpitch, cyaw)[:3, :3]
        T_camera_to_ground = np.eye(4)
        T_camera_to_ground[:3, :3] = self.R_camera
        T_camera_to_ground[:3, 3] = t_camera
        self.T_camera_to_ground = T_camera_to_ground 
    
    def _init_hand_transfer(self, ):
        self.left_hand_orientation = np.array([np.pi/2, np.pi, -np.pi/2])
        self.right_hand_orientation = np.array([np.pi/2, np.pi, -np.pi/2])
        self.R_left_hand = tf.transformations.euler_matrix(*self.left_hand_orientation)[:3, :3]
        self.R_right_hand = tf.transformations.euler_matrix(*self.right_hand_orientation)[:3, :3]
        self.R_left_hand_cam = tf.transformations.euler_matrix(0, 0, np.pi)[:3, :3]
        self.R_right_hand_cam = tf.transformations.euler_matrix(0, np.pi, np.pi)[:3, :3]
        
    
    def transfer_hand(self, hand_type, hand_angle):
        # hand_angle = [0,0,0]
        print(f"hand_angle: {hand_angle}")
        if hand_type == 'Left':
            R_hand = self.R_left_hand
            R_hand_cam = self.R_left_hand_cam
        elif hand_type == 'Right':
            hand_angle[2] = -(hand_angle[2] - np.pi/2) + np.pi/2
            
            R_hand = self.R_right_hand
            R_hand_cam = self.R_right_hand_cam
            R_hand_cam[1, 0] = -R_hand_cam[1, 0]
            # print(R_hand_cam)
        R_object = tf.transformations.euler_matrix(hand_angle[0], hand_angle[1], hand_angle[2])[:3, :3]
        
        # R_hand_ground = np.dot(R_hand, R_object)
        R_hand_ground = np.dot(R_object, R_hand)
        R_hand_ground = np.dot(R_hand_cam, R_hand_ground)
        R_hand_ground = np.dot(self.R_camera, R_hand_ground)
        p_hand_orientation = tf.transformations.euler_from_matrix(R_hand_ground)
        return np.array(p_hand_orientation)
            
    
    def transfer_point(self, point, ):
        return np.dot(self.T_camera_to_ground, np.concatenate((point, [1]), axis=0))[:3]
    
    def transfer_angle(self, angle):
        R_object = tf.transformations.euler_matrix(angle[0], angle[1], angle[2])[:3, :3]
        R_object_ground = np.dot(self.R_camera, R_object)
        p_ground_orientation = tf.transformations.euler_from_matrix(R_object_ground)
        return np.array(p_ground_orientation)
    
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
        
        return p_ground
        
    
    def get_joints(self, result, index, operator2mano):
        
        keypoint_3d = result.multi_hand_world_landmarks[index]
        keypoint_2d = result.multi_hand_landmarks[index]
        num_box = len(result.multi_hand_landmarks)

        # Parse 3d keypoints from MediaPipe hand detector
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ operator2mano
        return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot
        
    
    def process_result(self, results):
        hand_list = [handedness.ListFields()[0][1][0].label for handedness in results.multi_handedness]
        # print(hand_list)
        if self.selfie:
            hand_list = [self.inverse_hand_dict[hand] for hand in hand_list]
        # print(hand_list)
        
        if 'Left' in hand_list:
            index = hand_list.index('Left')
            joints = self.get_joints(results, index, self.operator2mano_left)
            self.left_hand.update_hand(results, index, joints)
            # print(f"Left grasp diatance: {self.left_hand.grasp_distance*100:.1f} cm")
            # print(f"Left angle: {self.left_hand.wrist_angle}")
            # print(f"joints:{self.left_hand.mediapipe_wrist_rot}")
        else:
            self.left_hand.update_none()
        
        if 'Right' in hand_list:
            index = hand_list.index('Right')
            joints = self.get_joints(results, index, self.operator2mano_right)
            self.right_hand.update_hand(results, index, joints)
            # print(f"Right grasp diatance:{self.right_hand.grasp_distance*100:.1f} cm")
            # print(f"Right angle: {self.right_hand.wrist_angle}")
        else:
            self.right_hand.update_none()
        
        
    
    def detect(self, bgr):
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(rgb)
        self.annotated_image = bgr
        if not results.multi_hand_landmarks:
            self.left_hand.update_none()
            self.right_hand.update_none()
            return False
        
        self.annotated_image = self.draw_landmarks_on_image(bgr, results)
        
        self.process_result(results)
        
        
        
        return True
    
    
    def get_hand_pose(self, point_cloud, depth_size):
        # point_cloud = point_cloud.copy()
        if self.left_hand.bool:
            left_wrist_image_coord = [int(self.left_hand.wrist_point_ic[0] * depth_size[0]), int(self.left_hand.wrist_point_ic[1] * depth_size[1])]
            if left_wrist_image_coord[0] >= depth_size[0] or left_wrist_image_coord[1] >= depth_size[1]:
                left_wrist_image_coord[0] = depth_size[0] - 1
                left_wrist_image_coord[1] = depth_size[1] - 1
            
            left_wirst_cam_coord = point_cloud[left_wrist_image_coord[1]][left_wrist_image_coord[0]]
            if left_wirst_cam_coord.all() == np.array([0, 0, 0]).all():
                logger.warning(f"Left Hand too close to the Cam.")
            else:
                self.left_hand.wirst_cam_coord = left_wirst_cam_coord
                self.left_hand.wirst_world_coord = self.transfer_point(left_wirst_cam_coord)
            print(f"LWrist_cc: {left_wirst_cam_coord}")
            
            
            R = np.array(self.left_hand.mediapipe_wrist_rot)
            
            roll, pitch, yaw = tf.transformations.euler_from_matrix(R)
            print(f"RPY: {roll, pitch, yaw}")
            
            self.left_hand.wirst_pose_angle = self.transfer_hand("Left", [roll, pitch, yaw])
            self.left_hand.wirst_pose = np.concatenate((self.left_hand.wirst_world_coord, self.left_hand.wirst_pose_angle))

            # 创建 4x4 的变换矩阵
            transform_matrix = tf.transformations.compose_matrix(
                scale=None,
                shear=None,
                angles= self.left_hand.wirst_pose_angle,
                # angles= [roll, pitch, yaw],
                translate= self.left_hand.wirst_world_coord,
                # quaternion=quat
            )
            self.left_hand.transform_matrix = transform_matrix
            
            
            
        if self.right_hand.bool:
            right_wrist_image_coord = [int(self.right_hand.wrist_point_ic[0] * depth_size[0]), int(self.right_hand.wrist_point_ic[1] * depth_size[1])]
            # print(f"LWrist: {left_wrist_image_coord}")
            if right_wrist_image_coord[0] >= depth_size[0] or right_wrist_image_coord[1] >= depth_size[1]:
                right_wrist_image_coord[0] = depth_size[0] - 1
                right_wrist_image_coord[1] = depth_size[1] - 1
            
            right_wirst_cam_coord = point_cloud[right_wrist_image_coord[1]][right_wrist_image_coord[0]]
            if right_wirst_cam_coord.all() == np.array([0, 0, 0]).all():
                logger.warning(f"Right Hand too close to the Cam.")
            else:
                self.right_hand.wirst_cam_coord = right_wirst_cam_coord
                self.right_hand.wirst_world_coord = self.transfer_point(right_wirst_cam_coord)
            print(f"RWrist_cc: {right_wirst_cam_coord}") 
            
            
            R = np.array(self.right_hand.mediapipe_wrist_rot)
            
            roll, pitch, yaw = tf.transformations.euler_from_matrix(R)
            print(f"RPY: {roll, pitch, yaw}")
            
            self.right_hand.wirst_pose_angle = self.transfer_hand("Right", [roll, pitch, yaw])
            self.right_hand.wirst_pose = np.concatenate((self.right_hand.wirst_world_coord, self.right_hand.wirst_pose_angle))

            # 创建 4x4 的变换矩阵
            transform_matrix = tf.transformations.compose_matrix(
                scale=None,
                shear=None,
                angles= self.right_hand.wirst_pose_angle,
                # angles= [roll, pitch, yaw],
                translate= self.right_hand.wirst_world_coord,
                # quaternion=quat
            )
            self.right_hand.transform_matrix = transform_matrix
            
    
    
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.multi_hand_landmarks
        handedness_list = detection_result.multi_handedness
        annotated_image = np.copy(rgb_image)
        

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            
            hand_landmarks = hand_landmarks_list[idx]
            label = detection_result.multi_handedness[idx].ListFields()[0][1][0].label
            if self.selfie:
                label = self.inverse_hand_dict[label]
            keypoint_2d = detection_result.multi_hand_landmarks[idx]

            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
            y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
            # x_coordinates = [hand_landmarks[i].x for i in range(21)]
            # y_coordinates = [hand_landmarks[i].y for i in range(21)]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{label}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image
        
        

# class HandDetection(object):
#     def __init__(self) -> None:
#         pass
    
#         self.left_detector = SingleHandDetector(hand_type="Right", selfie=False)
    
#     hand_detector = SingleHandDetector()
    # return hand_detector