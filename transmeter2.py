import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import pandas as pd
import socket
import json
mp_pose = mp.solutions.pose
# Server configuration
HOST = '127.0.0.1'
PORT = 65432

# Create a TCP/IP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the server's address and port
client_socket.connect((HOST, PORT))

def is_between(number, lower_limit, upper_limit):
    return lower_limit <= number <= upper_limit

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

def write_data_to_csv(csv_writer, frame, frame_count, elapsed_time,status):
    row = []

    for connection in mp_pose.POSE_CONNECTIONS:
        if connection[0] in selected_joint_ids and connection[1] in selected_joint_ids:
            start_point = landmarks[connection[0] - selected_joint_ids[0]]
            end_point = landmarks[connection[1] - selected_joint_ids[0]]
            
            # Calculate and display angle
            angle = calculate_angle(start_point, landmarks[selected_joint_ids.index(connection[0])], end_point)
            row.append(int(angle))

    # Add relative coordinates to the row
    row.extend(relative_landmarks)
    row.extend(str(status))

    # Write the row to CSV file
    csv_writer.writerow(row)

# Specify the folder containing video files
video_folder = './samples/Originals/E1/starts/'

# Get a list of all video files in the folder
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# Specify the joint names you want to track
selected_joint_names = ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
selected_joint_ids = [11, 13, 15, 12, 14, 16] 




    
    
if not video_files:
    print("No video files found in the specified folder.")
else:
    for video_file in video_files:
        # Open the video file
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        # Check if the video is opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_file}")
            continue

 
        # Check if the video is opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_file}")
            continue

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                # Capture frame-by-frame
                
                ret, frame = cap.read()
                # Check if the frame is read successfully
                if not ret:
                    print(f"Finished processing video: {video_file}")
                    break
                
           
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detection
                results = pose.process(image)
                
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                row = []
                # Extract and draw landmarks for selected joint IDs
                try:
                    landmarks = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                                    for i, landmark in enumerate(results.pose_landmarks.landmark)
                                    if i in selected_joint_ids]

                    # Calculate and print relative coordinates
                    relative_landmarks = calculate_relative_coordinates( frame.shape[:2], landmarks,results,frame)
                    # print(relative_landmarks)
                    # Create JSON object
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
                            'right_shoulder_left_shoulder': [row[2]],
                            'right_elbow_wrist_angle':[row[4]],
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



                    json_object = json.dumps(return_data, indent=4)
                    
                    # Print JSON object
                    print(json_object)
                
                            # Send a message
                    
                    client_socket.sendall(json_object.encode())
                    # Receive the random number from the server
                    random_number = client_socket.recv(1024).decode()
                    print(f"Received random number from server: {random_number}")
                    

                            


                except Exception as e:
                    print(f"Error processing video {video_file}: {e}")

                # Display the frame in a window
                cv2.imshow('Mediapipe Feed', image)
                
                # Break the loop when the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the video file
            cap.release()

# Close all windows
cv2.destroyAllWindows()



