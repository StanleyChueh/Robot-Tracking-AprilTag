import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion
import math
import tf2_ros

class AprilTagFollowing(Node):
    def __init__(self):
        super().__init__('apriltag_following')

        # Publisher for movement commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # TF Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters for tracking behavior
        self.linear_speed = 0.1     # Linear speed (m/s)
        self.angular_speed = 0.5    # Maximum angular speed (rad/s)
        self.x_threshold = 0.1      # Horizontal offset threshold for centering
        self.follow_distance = 0.6  # Desired distance from the AprilTag (meters)
        self.stale_timeout = 0.5    # Timeout in seconds for stale transforms

        # Frame names
        self.tag_frame = "tag36h11:2"
        self.camera_frame = "camera_color_optical_frame"
        self.robot_frame = "base_link"

        # State variable to track tag detection
        self.tag_detected = False

        self.get_logger().info("AprilTag following node initialized using TF transforms.")

        # Timer to call the follow_tag function periodically
        self.timer = self.create_timer(0.1, self.follow_tag)

    def follow_tag(self):
        try:
            # Lookup the transform from the camera to the AprilTag frame
            camera_to_tag = self.tf_buffer.lookup_transform(self.camera_frame, self.tag_frame, rclpy.time.Time())

            # Check if the transform is stale
            current_time = self.get_clock().now()
            transform_time = camera_to_tag.header.stamp.sec + camera_to_tag.header.stamp.nanosec * 1e-9
            time_diff = current_time.nanoseconds * 1e-9 - transform_time

            if time_diff > self.stale_timeout:
                self.get_logger().warn(f"Stale transform detected (time_diff: {time_diff:.2f}s). Stopping the robot.")
                self.stop_robot()
                self.tag_detected = False  # Update state to indicate tag is lost
                return

            # Update the state to indicate the tag is detected
            self.tag_detected = True
            self.get_logger().info(f"Transform found: {camera_to_tag}")

            # Process the transform if it exists
            position = camera_to_tag.transform.translation
            orientation = camera_to_tag.transform.rotation

            # Calculate distance and yaw
            distance = math.sqrt(position.x ** 2 + position.y ** 2 + position.z ** 2)
            _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

            self.get_logger().info(f"Distance to AprilTag: {distance:.2f} m, Horizontal Offset: {position.x:.2f} m, Yaw angle: {yaw:.2f} rad")

            # Initialize the twist command
            twist = Twist()

            # Forward motion logic
            if distance > self.follow_distance:
                twist.linear.x = self.linear_speed
            else:
                twist.linear.x = 0.0  # Stop forward movement if close enough

            # Horizontal centering logic based on x-position of tag in camera frame
            if abs(position.x) > self.x_threshold:
                # Scale angular velocity based on horizontal offset
                twist.angular.z = -self.angular_speed * position.x
                self.get_logger().info(f"Adjusting position: {'Right' if position.x > 0 else 'Left'}")
            else:
                twist.angular.z = 0.0  # Stop turning if centered

            # Publish the velocity command
            self.cmd_vel_publisher.publish(twist)

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            # Handle loss of AprilTag frame
            if self.tag_detected:
                self.get_logger().warn(f"AprilTag frame {self.tag_frame} not found. Stopping the robot.")
                self.stop_robot()
                self.tag_detected = False  # Update state to indicate tag is lost

    def stop_robot(self):
        """Stops the robot by publishing zero velocities."""
        stop_msg = Twist()  # Twist message with all zeros (default)
        self.cmd_vel_publisher.publish(stop_msg)
        self.get_logger().info("Robot stopped.")

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagFollowing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

