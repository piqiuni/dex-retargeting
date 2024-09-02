import ctypes
import time
from depthcam_sdk import yd_people_sensor as yd
import cv2
import math
import numpy as np

class DepthCam(object):
    def __init__(self) -> None:
        pass
        count = yd.c_uint()
        error_code = yd.Sensor.get_count(count)
        if yd.ErrorCode.success.value != error_code:
            print("Failed to get sensor count with error code: %s" % error_code)
            exit(1)

        self.sensor = yd.Sensor()
        reslutions = [yd.ColorResolution.vga, yd.DepthResolution.vga]
        reslutions_size = [(320, 240), (640,480), (1280, 960)]
        error_code = self.sensor.initialize(reslutions[0], reslutions[1], yd.c_bool(True), yd.c_uint(0))
        if yd.ErrorCode.success.value != error_code:
            print("Failed to initialize sensor with error code: %s" % error_code)
            exit(1)

        error_code = self.sensor.set_depth_mapped_to_color(True)
        if not self.sensor.is_depth_mapped_to_color():
            print("Failed to set depth mapped to color")
            exit(1)

        error_code = self.sensor.set_near_mode(True)
        if not self.sensor.is_near_mode:
            print("Failed to set near mode")
            exit(1)

        error_code = self.sensor.start()
        if yd.ErrorCode.success.value != error_code:
            print("Failed to start sensor with error code: %s" % error_code)
            exit(1)
            
        
        self.color_frame = yd.ColorFrame()
        self.depth_frame = yd.DepthFrame()
        self.publish_data = yd.PublishData()
        
        self.depth_range_x = [85, 590] #(85,35)  (590, 429)
        self.depth_range_y = [35, 429]
        
        # self.color_image = None
        # self.depth_image = None
        
        self.color_size = reslutions_size[reslutions[0].value]
        color_length = self.color_size[0] * self.color_size[1] * 4
        self.color_array_type = yd.c_char * color_length
        
        self.depth_size = reslutions_size[reslutions[1].value]
        depth_length = self.depth_size[0] * self.depth_size[1]
        self.depth_array_type = yd.c_ushort * depth_length
        
    def get_frames(self, ):
        self.start_time = time.time()
        error_code = self.sensor.get_color_frame(self.color_frame)
        if yd.ErrorCode.success.value != error_code:
            return False

        error_code = self.sensor.get_depth_frame(self.depth_frame)
        if yd.ErrorCode.success.value != error_code:
            return False

        error_code = self.sensor.get_publish_data(self.publish_data)
        if yd.ErrorCode.success.value != error_code:
            return False
        
        color_addr = yd.addressof(self.color_frame.pixels.contents)
        self.color_image = np.frombuffer(self.color_array_type.from_address(color_addr), dtype=np.uint8).reshape(self.color_frame.height, self.color_frame.width, 4)
        self.color_image = np.fliplr(self.color_image)  # 左右翻转
        
        depth_addr = yd.addressof(self.depth_frame.pixels.contents)
        self.depth_image = np.frombuffer(self.depth_array_type.from_address(depth_addr), dtype=np.uint16).reshape(self.depth_frame.height, self.depth_frame.width)
        self.depth_image = np.fliplr(self.depth_image)  # 左右翻转
        return True
    
    def get_point_cloud(self, ):
        cloudframe = yd.PointCloudFrame()
        cloud_length = self.depth_frame.width * self.depth_frame.height * 3
        buffer = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_float) * cloud_length)
        cloudframe.point = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_float))
        
        self.sensor.convert_depth_frame_to_point_cloud(self.depth_frame.width, self.depth_frame.height, self.depth_frame.pixels.contents, cloudframe.point.contents)
        
        cloud_array_type = yd.c_float * cloud_length
        addr = ctypes.addressof(cloudframe.point.contents)
        point_cloud = np.frombuffer(cloud_array_type.from_address(addr), dtype=np.float32).reshape(self.depth_frame.height, self.depth_frame.width, 3)
        self.point_cloud = np.fliplr(point_cloud).copy()
        self.point_cloud_value = self.point_cloud[:, :, 2]
        
        # print(f"=++++={self.point_cloud[240][320]}")
        
    def get_point_could_image(self):
        import matplotlib.pyplot as plt
        fig=plt.figure()
        ax = plt.axes()
        im = ax.imshow(self.point_cloud_value, vmin=0, vmax=5, cmap='jet')
        # 修改0.01可改变图像与颜色条的距离，修改0.02可改变颜色条自己的宽度
        cax = fig.add_axes([ax.get_position().x1+0.015,ax.get_position().y0,0.02,ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        # 设置 colorbar 的标签
        cbar.set_label('Depth (m)', rotation=90, labelpad=20)
        #设置xy轴标签
        # ax.set_xlabel("x/pixel",fontsize=12.5)
        # ax.set_ylabel("y/pixel",fontsize=12.5)
        # plt.show()
        
        # 将 Matplotlib 图像转换为 OpenCV 格式
        fig.canvas.draw()
        colorbar_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        colorbar_img = colorbar_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        colorbar_img = cv2.cvtColor(colorbar_img, cv2.COLOR_RGB2BGR)
        width = self.color_size[1]
        colorbar_img = cv2.resize(colorbar_img, (int(colorbar_img.shape[1] * width / colorbar_img.shape[0]), width), interpolation=cv2.INTER_AREA)
        fig.clf()
        plt.close()
        return colorbar_img
    
        
    def disp_iamges(self, annotated_image):
        
        
        return self.color_image, self.depth_image