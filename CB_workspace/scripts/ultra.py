#!/usr/bin/env python3
## Vision
from ultralytics import YOLO
import pyrealsense2 as rs

## ROS
import rospy
from std_msgs.msg import String

#Self-defined msg
from yolo.msg import yolomsg
from yolo.srv import RD_signal, RD_signalResponse

## Other tools
import math
import numpy as np
from threading import Thread, Event

# Define variables
DEVICE_SERIAL = "949122070619" # Realsense device serial number
XOFFSET = 0.014382474 # x distance with CB center
ZOFFSET = 0.099894668 # Z distance with CB center
THETA = 0 # camera depression angle in degree
WIN_WIDTH, WIN_HEIGHT = 640, 480 # camera resolution
VERBOSE = True # YOLO verbose (showing detection ouput)
WAITING_TIME = 0 # waiting time for first publish

class Node:
    def __init__(self):
        rospy.init_node('CB_Server')

        ### YOLO model ###
        self.model = YOLO("src/yolo/weight/onboard_cam.pt")
        #self.model.fuse() # Fuse for speed

        ### Publisher ###
        #CB detection topic
        self.pub = rospy.Publisher('/robot/objects/global_info', String, queue_size=10)
        #Ready signal service
        self.service = rospy.Service('/robot/startup/ready_signal', RD_signal, self.ready_callback)
        
        rospy.loginfo("CB waiting for Ready Signal...")
        self.publish_thread = None
        self.first_publish_event = Event()

    def ready_callback(self, request) -> RD_signalResponse:
        ready = request.ready
        if ready:
            rospy.loginfo("Received ready signal. Starting CB detection...")
            if self.publish_thread is None or self.publish_thread.is_alive() is False:
                self.first_publish_event.clear()

                #Open yolo thread
                self.publish_thread = Thread(target=self.yolo)
                self.publish_thread.start()

                # Wait for the first publish to complete
                self.first_publish_event.wait()

            success = True
        else:
            rospy.logwarn("Received ready signal is False. Aborting process...")
            success = False

        return RD_signalResponse(success)

    def yolo(self):
        while rospy.is_shutdown() is False:
            color_img, depth_img = realsense_class.wait_for_frames()
            background_rm_img = self.background_removal(color_img, depth_img)

            #YOLO 
            results = self.model(source = background_rm_img, verbose = VERBOSE)

            plantmsg = yolomsg()
            plantmsg.x = []
            plantmsg.y = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x, y = int((x1 + x2) / 2), int(y1 / 4 + y2 * 3 / 4)
                    depth = depth_img.get_distance(x, y)

                    Xtarget, Ztarget = self.transform_coordinates(x, y, depth)
                    plantmsg.x.append(Xtarget)
                    plantmsg.y.append(Ztarget)
            
            if self.first_publish_event.is_set() is False:
                rospy.sleep(WAITING_TIME)
                rospy.loginfo("Processing YOLO...")
                self.first_publish_event.set()

            self.pub.publish("plantmsg")

    def background_removal(self, color_img, depth_img) -> np.ndarray:
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_img.get_data())
        depth_image = np.asanyarray(depth_img.get_data())
        # Stack depth image to 3 channels
        depth_image_3 = np.dstack((depth_image, depth_image, depth_image))
        # Remove background 
        background_rm_img = np.where((depth_image_3 > depth_img.get_distance(0,0) + 0.2) | (depth_image_3 <= 0), 153, color_image)
        return background_rm_img

    def transform_coordinates(self, x, y, depth):
        Xtemp = depth * (x - rs.intr.ppx) / rs.intr.fx
        Ytemp = depth * (y - rs.intr.ppy) / rs.intr.fy
        Ztemp = depth
        Xtarget = 0
        Ytarget = 0
        Ztarget = Ztemp*math.cos(math.radians(THETA))
        return Xtarget, Ztarget
        
class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        # Camera configuration
        self.config = rs.config()
        self.config.enable_device(DEVICE_SERIAL)
        self.config.enable_stream(rs.stream.color, WIN_WIDTH, WIN_HEIGHT, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, WIN_WIDTH, WIN_HEIGHT, rs.format.z16, 30)
        # Streaming configuration
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def wait_for_frames(self):
        aligned_frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        return color_frame, depth_frame
    
if __name__ == '__main__':
    try:
        vision_node = Node()
        realsense_class = RealsenseCamera()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
