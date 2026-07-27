import argparse
import imutils
import time
import cv2
import sys
from imutils.video import VideoStream

# 1. Setup the Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_5X5_100",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

# 2. Define the ArUco dictionary mapping
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# 3. Verify the dictionary exists and load it
if args["type"] not in ARUCO_DICT:
	print(f"[INFO] ArUCo tag of '{args['type']}' is not supported")
	sys.exit(0)

print("[INFO] detecting '{}' tags...".format(args["type"]))
# --- UPDATED FOR OPENCV 4.7+ ---
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# 4. Start the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 5. The Main Loop
while True:
	# Grab the frame and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)
	height, width, channels = frame.shape

	# Find the exact center of the screen
	screen_cX = int(width / 2.0)
	screen_cY = int(height / 2.0)

	# --- UPDATED FOR OPENCV 4.7+ ---
	# Detect ArUco markers using the new detector object
	(corners, ids, rejected) = detector.detectMarkers(frame)

	# Verify *at least* one ArUco marker was detected
	if len(corners) > 0:
		# Flatten the ArUco IDs list
		ids = ids.flatten()

		# Loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# Extract the marker corners
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			# Convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			# Compute and draw the center (x, y)-coordinates of the ArUco marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

			# --- MISSION 1: THE TARGET PAD VALIDATOR ---
			if markerID == 24:
				# Draw in GREEN for our target drone pad
				cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
				cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
				cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
				cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
				
				# --- MISSION 2: THE TARGETING LASER ---
				cv2.line(frame, (screen_cX, screen_cY), (cX, cY), (255, 0, 0), 2)

				# --- MISSION 3: HUD ALERT ---
				cv2.putText(frame, "STATUS: TARGET ACQUIRED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

				# --- THE STATE MACHINE VARIABLES ---
				# Check if within the 50px deadzone on BOTH axes
				is_centered = (abs(cX - screen_cX) <= 50) and (abs(cY - screen_cY) <= 50)
				
				# Check if tilt is within the 30px deadzone
				tilt = abs(topLeft[1] - topRight[1])
				is_yaw_locked = (tilt <= 30)
				
				# Measure altitude/distance
				marker_width = topRight[0] - topLeft[0]

				# --- MISSION 7: MASTER FLIGHT CONTROLLER ---
				# PRIORITY 1: Center the drone (X/Y)
				if not is_centered:
					if (cX < screen_cX - 50):
						cv2.putText(frame, "COMMAND: FLY LEFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
					elif (cX > screen_cX + 50):
						cv2.putText(frame, "COMMAND: FLY RIGHT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
					elif (cY < screen_cY - 50):
						cv2.putText(frame, "COMMAND: FLY FORWARD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
					elif (cY > screen_cY + 50):
						cv2.putText(frame, "COMMAND: FLY BACKWARD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
				
				# PRIORITY 2: Align the Yaw (Rotation)
				elif not is_yaw_locked:
					cv2.putText(frame, "COMMAND: YAW ALIGNING...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
				
				# PRIORITY 3: Descend and Land (Altitude)
				else:
					if marker_width >= 200:
						cv2.putText(frame, "COMMAND: READY TO LAND", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
					else:
						cv2.putText(frame, "COMMAND: DESCENDING...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

			else:
				# Draw in RED for everything else (Wrong Landing Pad!)
				cv2.line(frame, topLeft, topRight, (0, 0, 255), 2)
				cv2.line(frame, topRight, bottomRight, (0, 0, 255), 2)
				cv2.line(frame, bottomRight, bottomLeft, (0, 0, 255), 2)
				cv2.line(frame, bottomLeft, topLeft, (0, 0, 255), 2)

			# Draw the ArUco marker ID text just above the marker
			cv2.putText(frame, str(markerID),
				(topLeft[0], topLeft[1] - 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
				
	else:
		# --- MISSION 3: NO TARGET FOUND HUD ---
		cv2.putText(frame, "STATUS: SEARCHING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
	
	# Show the output frame on your screen
	cv2.imshow("Drone HUD", frame)
	key = cv2.waitKey(1) & 0xFF

	# If the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# 6. Cleanup after breaking the loop
cv2.destroyAllWindows()
vs.stop()