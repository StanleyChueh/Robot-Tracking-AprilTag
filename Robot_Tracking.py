import cv2
import numpy as np
from apriltag import apriltag
import rclpy
from geometry_msgs.msg import Twist

def calculate_center(rect):
    center_x = int((rect[0][0][0] + rect[2][0][0]) / 2)
    center_y = int((rect[0][0][1] + rect[2][0][1]) / 2)
    return center_x, center_y

def estimate_distance(apparent_width, tag_size, focal_length):
    return (tag_size * focal_length) / apparent_width

def draw_apriltag(frame, detection, tag_size, focal_length):
    tag_id = detection["id"]
    rect = np.array(detection["lb-rb-rt-lt"], dtype=np.int32).reshape((-1, 1, 2))

    center = calculate_center(rect)
    apparent_width = abs(rect[0][0][0] - rect[1][0][0])
    distance = estimate_distance(apparent_width, tag_size, focal_length)

    cv2.polylines(frame, [rect], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(frame, f"ID: {tag_id}", (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Dist: {distance:.2f}m", (center[0] - 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def april_tag_tracking():
    focal_length = 3.67  # Focal length in mm
    tag_size = 0.16  # Tag size in meters
    twist = Twist()

    detector = apriltag("tag36h11")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (160, 120))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)
        
        for detection in detections:
            tag_id = detection["id"]
            rect = np.array(detection["lb-rb-rt-lt"], dtype=np.int32).reshape((-1, 1, 2))
            center = calculate_center(rect)

            if center[0] < 80:
                print("Turning Left!")
                twist.angular.z = 0.5
            elif 80 <= center[0] <= 89:
                print("Going Forward!")
                twist.linear.x = 0.1
            elif center[0] > 89:
                print("Turning Right!")
                twist.angular.z = -0.5

            draw_apriltag(frame, detection, tag_size, focal_length)

        cv2.imshow("AprilTag Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Reset twist after processing each frame
        twist.linear.x = 0.0
        twist.angular.z = 0.0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    april_tag_tracking()
