import numpy as np
from Depthcam import DepthCam
from HandDetection import MultiHandDetector
import time
import cv2


if __name__ == "__main__":
    cam = DepthCam()
    detector = MultiHandDetector(selfie = True)
    
    try:
        time_now = time.time()
        while True:
            if not cam.get_frames():
                continue
            
            
            bgr_image = cv2.cvtColor(cam.color_image, cv2.COLOR_BGRA2BGR)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cam.depth_image, alpha=0.03), cv2.COLORMAP_JET)

            got_hand = detector.detect(bgr_image)
            
            
            cam.get_point_cloud()
            colorbar_img = cam.get_point_could_image()
            
            if got_hand:
                detector.get_hand_pose(cam.point_cloud, cam.depth_size)
            

            # Get hand pos
            
            # flipped_image  = np.fliplr(self.color_image)
            
            # detector.detect(srgb_image)
            
 
            images = np.hstack((detector.annotated_image, colorbar_img))
            
            frame_rate = 1/(time.time()-time_now)
            frame_rate = int(frame_rate*10)/10
            # print(f"frame_rate: {frame_rate}")
            time_now = time.time()
            cv2.putText(images, str(int(frame_rate*10)/10), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.namedWindow('YDViewer', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('YDViewer', images)

            key = cv2.waitKey(1)
            if key in (27, ord("q")):
                break
    finally:
        cam.sensor.stop()
        cam.sensor.uninitialize()