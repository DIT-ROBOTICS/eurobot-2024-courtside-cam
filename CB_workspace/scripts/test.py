#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from yolo.srv import RD_signal, RD_signalResponse
from threading import Thread, Event

class ReadyPublisher:
    def __init__(self):
        rospy.init_node('ready_server')
        self.pub = rospy.Publisher('/hello', String, queue_size=10)
        self.service = rospy.Service('ready_service', RD_signal, self.ready_callback)
        rospy.loginfo("Ready to receive ready requests.")
        self.publish_thread = None
        self.first_publish_event = Event()

    def ready_callback(self, request):
        ready = request.ready
        if ready:
            rospy.loginfo("Received ready signal. Starting to publish 'hello world'...")
            if self.publish_thread is None or self.publish_thread.is_alive() is False:
                self.first_publish_event.clear()

                #Open yolo thread
                self.publish_thread = Thread(target=self.publish_hello_world)
                self.publish_thread.start()

                # Wait for the first publish to complete
                self.first_publish_event.wait()

            success = True
        else:
            rospy.logwarn("Received ready signal is False. Aborting process...")
            success = False

        return RD_signalResponse(success)

    def publish_hello_world(self):

        while rospy.is_shutdown() is False:

            if self.first_publish_event.is_set() is False:
                rospy.sleep(5)
                rospy.loginfo("First time publishing 'hello world'...")
                self.first_publish_event.set()

            self.pub.publish("hello world")

if __name__ == '__main__':
    try:
        ready_publisher = ReadyPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
