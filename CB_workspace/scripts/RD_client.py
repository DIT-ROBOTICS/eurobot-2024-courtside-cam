#!/usr/bin/env python3

import rospy
from yolo.srv import RD_signal

def ready_client(ready):
    rospy.wait_for_service('/robot/startup/ready_signal')
    try:
        ready_service = rospy.ServiceProxy('/robot/startup/ready_signal', RD_signal)
        response = ready_service(ready)
        return response.success
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)

if __name__ == '__main__':
    rospy.init_node('ready_client')
    
    ready = True
    success = ready_client(ready)
    if success:
        rospy.loginfo("Server successfully Running.")
    else:
        rospy.logwarn("Server failed to receive ready signal.")
