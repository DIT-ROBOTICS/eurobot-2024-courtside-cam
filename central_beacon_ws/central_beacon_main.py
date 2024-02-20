#!/usr/bin/python3
import cv2
import math
import numpy as np
import pyrealsense2 as rs
import rospy
from yolo.msg import existancemsg
from yolo.msg import positionmsg
from ultralytics import YOLO

xoffset = 0 
zoffset = 0
THETA = 0
WIN_WIDTH, WIN_HEIGHT = 640, 480

class RectangularArea:
    def __init__(self, name, x_start, x_end, y_start, y_end):
        self.name = name
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end

    def is_inside(self, x, y):
        return self.x_start <= x <= self.x_end and self.y_start <= y <= self.y_end

class AreaChecker:
    def __init__(self):
        self.areas = [
            RectangularArea(0, 0, 5, 0, 5),
            RectangularArea(1, 5, 10, 0, 5),
            RectangularArea(2, 10, 15, 0, 5),
            RectangularArea(3, 0, 5, 5, 10),
            RectangularArea(4, 5, 10, 5, 10),
            RectangularArea(5, 10, 15, 5, 10), # need to decide the six initial area
        ]

    def find_area(self, x, y):
        for area in self.areas:
            if area.is_inside(x, y):
                return area.name
        return 6 

class YoloDetector:
    def __init__(self):
        self.model = YOLO("src/yolo/weight/CB1.pt")
        self.model.fuse()
        self.class_names = ("plant")

    def detect_objects(self, img):
        results = self.model(img, stream=True)
        return results

    def draw_bounding_boxes(self, img, results, depth_frame, intr, theta):
        
        self.init_msg()
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x, y = int((x1 + x2) / 2), int(y1 / 4 + y2 * 3 / 4)
                depth = depth_frame.get_distance(x, y)
                AREA = AreaChecker.find_area(x, y)
                Xtarget, Ztarget = self.transform_coordinates(depth, x, y, intr, theta)

                if AREA != 6:
                    self.existmsg.amount[AREA] += 1                
                    self.inposemsg.x.append(Xtarget)
                    self.inposemsg.y.append(Ztarget)
                else:
                    self.outposemsg.x.append(Xtarget)
                    self.outposemsg.y.append(Ztarget)

                self.cv2_draw(img, x, y, x1, y1, x2, y2, Xtarget, Ztarget)
        
        exist_pub.publish(self.existmsg)
        inside_pub.publish(self.inposemsg)
        outside_pub.publish(self.outposemsg)
        
        return img
    
    def init_msg(self):
        existmsg = existancemsg()
        inposemsg = positionmsg()
        outposemsg = positionmsg()
        existmsg.existance = [0] * 6
        existmsg.amount = [0] * 6
        inposemsg.x = []
        inposemsg.y = []
        outposemsg.x = []
        outposemsg.y = []

    def transform_coordinates(self, depth, x, y, intr, theta):
        Xtemp = depth * (x - intr.ppx) / intr.fx
        Ytemp = depth * (y - intr.ppy) / intr.fy
        Ztemp = depth

        Xtarget = Xtemp + xoffset
        Ztarget = Ztemp*math.cos(math.radians(theta)) + zoffset

        return Xtarget, Ztarget

    def cv2_draw(self, img, x, y, x1, y1, x2, y2, Xtarget, Ztarget):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.circle(img, (x, y), 3, (0, 0, 255), 2)
        cv2.putText(img, "({:.3f}, {:.3f})".format(Xtarget, Ztarget), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #self.config.enable_device('949122070603')
        self.config.enable_stream(rs.stream.color, WIN_WIDTH, WIN_HEIGHT, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, WIN_WIDTH, WIN_HEIGHT, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def wait_for_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        return color_frame, depth_frame

if __name__ == "__main__":
    rospy.init_node("central_beacon_main")
    exist_pub = rospy.Publisher("plant_existance", existancemsg, queue_size=10)
    inside_pub = rospy.Publisher("in_plant", positionmsg, queue_size=10)
    outside_pub = rospy.Publisher("out_plant", positionmsg, queue_size=10)

    yolo_detector = YoloDetector()
    realsense_camera = RealsenseCamera()

    cv2.namedWindow('RealSense YOLO', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RealSense YOLO', WIN_WIDTH, WIN_HEIGHT)

    try:
        while rospy.is_shutdown() == False:
            color_frame, depth_frame = realsense_camera.wait_for_frames()
            img = cv2.resize(np.asanyarray(color_frame.get_data()), (WIN_WIDTH, WIN_HEIGHT))

            results = yolo_detector.detect_objects(img)
            img_with_boxes = yolo_detector.draw_bounding_boxes(img, results, depth_frame, realsense_camera.intr, THETA)

            cv2.imshow('RealSense YOLO', img_with_boxes)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        realsense_camera.pipeline.stop()
        cv2.destroyAllWindows()