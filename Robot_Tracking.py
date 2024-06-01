import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import time
from apriltag import apriltag

# Dictionary to store stop times for each tag_id
tag_stop_times = {}  

def calculate_center(rect):
    center_x = int((rect[0][0][0] + rect[2][0][0]) / 2)
    center_y = int((rect[0][0][1] + rect[2][0][1]) / 2)
    return center_x, center_y

def estimate_distance(apparent_width, tag_size, focal_length):
    distance = (tag_size * focal_length) / apparent_width
    return distance

def draw_apriltag(frame, detection, tag_size, focal_length):
    tag_id = detection["id"]
    rect = detection["lb-rb-rt-lt"]
    rect = np.array(rect, dtype=np.int32).reshape((-1, 1, 2))

    center = calculate_center(rect)
    apparent_width = abs(rect[0][0][0] - rect[1][0][0])
    distance = estimate_distance(apparent_width, tag_size, focal_length)

    cv2.polylines(frame, [rect], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(frame, f"ID: {tag_id}", (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)

def stop_callback(timer, twist, tag_id, cmd_vel_pub):
    # Stop callback, stop the robot or take appropriate action
    print(f"Stopping for three seconds for AprilTag {tag_id}...")
    twist.linear.x = 0.0  # Stop linear movement
    twist.angular.z = 0.0  # Stop angular movement
    cmd_vel_pub.publish(twist)

    time.sleep(3)

def april_tag_callback(frame, detector, tag_size, focal_length, expected_tag_id, node, cmd_vel_pub):
    global tag_stop_times

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    
    if not detections:
        # No AprilTag detected, return None (empty)
        return None, expected_tag_id

    twist = Twist()
    correct_tag_detected = False  # Flag to check if the correct tag is detected

    for detection in detections:
        tag_id = detection["id"]
        rect = detection["lb-rb-rt-lt"]
        rect = np.array(rect, dtype=np.int32).reshape((-1, 1, 2))

        draw_apriltag(frame, detection, tag_size, focal_length)  # Draw bounding box and display ID

        center = calculate_center(rect)
        apparent_width = abs(rect[0][0][0] - rect[1][0][0])
        distance = estimate_distance(apparent_width, tag_size, focal_length)

        if tag_id == expected_tag_id and tag_id not in tag_stop_times:
            correct_tag_detected = True  # Set the flag if the correct tag is detected

            if distance < 0.45:
                print(f"Stopping for three seconds for AprilTag {tag_id}...")
                tag_stop_times[tag_id] = True  # Mark the tag as printed

                current_time = time.time()
                tag_stop_times[str(tag_id) + "_time"] = current_time  # Update stop time

                stop_timer = node.create_timer(3.0, lambda timer: stop_callback(timer, twist, tag_id, cmd_vel_pub))

                print(f"Detected AprilTag {tag_id}, distance: {distance}")
                return twist, expected_tag_id + 1  # Move to the next expected tag ID

            if center[0] < 80:
                print("Turning Left!")
                twist.angular.z = 0.5
            elif 80 <= center[0] <= 89:
                print("Going Forward!")
                twist.linear.x = 0.1
            elif center[0] > 89:
                print("Turning Right!")
                twist.angular.z = -0.5

    # If the correct tag is not detected, stop publishing cmd_vel
    if not correct_tag_detected:
        return None, expected_tag_id

    # If no AprilTag triggered any return, return the twist without publishing cmd_vel
    return twist, expected_tag_id

def main():
    global tag_stop_times

    focal_length = 3.67  
    tag_size = 0.16  
    expected_tag_id = 0

    detector = apriltag("tag36h11")

    rclpy.init()
    node = rclpy.create_node('apriltag_follower')
    cmd_vel_pub = node.create_publisher(Twist, 'cmd_vel', 10)
    print("Publisher created successfully")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    while rclpy.ok():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (160, 120))

        twist_apriltag, expected_tag_id = april_tag_callback(frame, detector, tag_size, focal_length, expected_tag_id, node, cmd_vel_pub)

        if twist_apriltag is not None:
            cmd_vel_pub.publish(twist_apriltag)

        cv2.imshow("AprilTag Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
