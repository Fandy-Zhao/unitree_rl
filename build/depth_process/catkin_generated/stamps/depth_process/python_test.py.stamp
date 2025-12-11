#!/usr/bin/env python3

import rospy
from std_msgs.msg import String


def python_node1():
    rospy.init_node('python_test')
    pub = rospy.Publisher('chatter', String, queue_size=10)
    

    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub.publish("Hello from Python")
        rate.sleep()

if __name__ == '__main__':
    python_node1()