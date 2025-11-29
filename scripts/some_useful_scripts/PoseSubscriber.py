import rospy
from geometry_msgs.msg import PoseStamped

class PoseSubscriber:
    def __init__(self):
        rospy.init_node('pose_listener', anonymous=False)
        self.subscriber = rospy.Subscriber("/phantom/phantom/pose", PoseStamped, self.callback)

    def callback(self, data):
        print("Received PoseStamped:")
        print(f"Position: x={data.pose.position.x}, y={data.pose.position.y}, z={data.pose.position.z}")
        print(f"Orientation: x={data.pose.orientation.x}, y={data.pose.orientation.y}, z={data.pose.orientation.z}, w={data.pose.orientation.w}")
        rospy.loginfo("Received PoseStamped:")
        rospy.loginfo("Position: x=%.2f, y=%.2f, z=%.2f", 
                      data.pose.position.x, data.pose.position.y, data.pose.position.z)
        rospy.loginfo("Orientation: x=%.2f, y=%.2f, z=%.2f, w=%.2f", 
                      data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w)

    def listen(self):
        rospy.spin()

if __name__ == '__main__':
    pose_subscriber = PoseSubscriber()
    pose_subscriber.listen()
