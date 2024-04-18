import cv2
import mediapipe as mp
import socket
import json
import numpy as np

# Server configuration
HOST = '127.0.0.1'
PORT = 65432

# Create a TCP/IP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the server's address and port
client_socket.connect((HOST, PORT))

# Define the selected joint IDs
selected_joint_ids = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                      mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                      mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                      mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                      mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                      mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
mp_pose = mp.solutions.pose
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_relative_coordinates( frame_shape,landmarks,results,frame):
    # Inside the loop where you process each frame and extract landmarks:
    try:
        nose_landmark = None
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i == mp_pose.PoseLandmark.NOSE:
                nose_landmark = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                break

        if nose_landmark is None:
            print("Nose landmark not found!")

        nose_x, nose_y = nose_landmark
        relative_landmarks = []
        for landmark in landmarks:
            x_relative = (landmark[0] - nose_x) / (frame.shape[1] - nose_x)
            y_relative = (landmark[1] - nose_y) / (frame.shape[0] - nose_y)
            relative_landmarks.extend([x_relative, y_relative])

        return relative_landmarks
    except Exception as e:
        print (e)

# Function to detect human poses using Mediapipe
def detect_poses(frame):
    
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Process the image
    results = pose.process(image_rgb)
    
    # Check if any pose landmarks are detected
    if results.pose_landmarks:
        joint_coordinates = {}
        for joint_id in selected_joint_ids:
            joint = results.pose_landmarks.landmark[joint_id]
            joint_coordinates[joint_id] = (int(joint.x * frame.shape[1]), int(joint.y * frame.shape[0]))
            # Draw a circle at the joint position
            cv2.circle(frame, joint_coordinates[joint_id], 5, (0, 255, 0), -1)
    
    row = []

    

    landmarks = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                                     for i, landmark in enumerate(results.pose_landmarks.landmark)
                                     if i in selected_joint_ids]
    relative_landmarks = calculate_relative_coordinates( frame.shape[:2], landmarks,results,frame)
    
    for connection in mp_pose.POSE_CONNECTIONS:
        if connection[0] in selected_joint_ids and connection[1] in selected_joint_ids:
            start_point = landmarks[connection[0] - selected_joint_ids[0]]
            end_point = landmarks[connection[1] - selected_joint_ids[0]]
            
            # Calculate and display angle
            angle = calculate_angle(start_point, landmarks[selected_joint_ids.index(connection[0])], end_point)
            row.append(int(angle))

    # print(row,relative_landmarks)
    return_data = {
            'left_shoulder_elbow_angle':[ row[0]],
            'left_elbow_wrist_angle': [row[1]],
            'right_shoulder_elbow_angle': [row[2]],
            'right_elbow_wrist_angle':[row[3]],
            'left_shoulder_x_rel': [relative_landmarks[0]],
            'left_shoulder_y_rel':  [relative_landmarks[1]],
            'left_elbow_x_rel':  [relative_landmarks[2]],
            'left_elbow_y_rel': [ relative_landmarks[3]],
            'left_wrist_x_rel':  [relative_landmarks[4]],
            'left_wrist_y_rel':  [relative_landmarks[5]],
            'right_shoulder_x_rel':  [relative_landmarks[6]],
            'right_shoulder_y_rel':  [relative_landmarks[7]],
            'right_elbow_x_rel': [relative_landmarks[8]],
            'right_elbow_y_rel': [ relative_landmarks[9]],
            'right_wrist_x_rel':  [relative_landmarks[10]],
            'right_wrist_y_rel': [ relative_landmarks[11]],
        }


    return frame, return_data

# Function to process video from webcam
def process_webcam():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, joint_coordinates = detect_poses(frame)
        
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to process video from file
def process_video_file(filename):
    try:
        # print (filename)
        cap = cv2.VideoCapture(filename)
        print (cap.isOpened())
        while cap.isOpened():
           
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, relative_landmarks = detect_poses(frame)
            
            # Create JSON object
            json_object = json.dumps(relative_landmarks, indent=4)
            
            # Print JSON object
            print(json_object)
        
                    # Send a message
            
            client_socket.sendall(json_object.encode())
            # Receive the random number from the server
            random_number = client_socket.recv(1024).decode()
            print(f"Received random number from server: {random_number}")
            

            cv2.imshow('Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print (e)

# Main function
def main():
    print("Choose video source:")
    print("1. Webcam")
    print("2. Video file")

    # choice = int(input("Enter your choice: "))
    choice = 2
    if choice == 1:
        process_webcam()
    elif choice == 2:
        filename = "./samples/Originals/E1/D1/Nested Sequence 01.mp4"
        process_video_file(filename)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
