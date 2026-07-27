import cv2
import cv2.aruco as aruco

# Initialize the camera (0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Define the dictionary (Must match the one used to generate the marker)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

print("Looking for markers... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab camera frame.")
        break

    # Convert to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # If at least one marker is found
    if ids is not None:
        # Draw the bounding box and ID on the video frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Print the detected ID to the terminal
        print(f"Drone Vision System - Detected Marker ID: {ids[0][0]}")

    # Display the video feed
    cv2.imshow('Drone ArUco Vision Test', frame)

    # Hit 'q' on the keyboard to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources when done
cap.release()
cv2.destroyAllWindows()