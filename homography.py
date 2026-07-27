import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import time

# 1. CREATE THE "VIRTUAL" IMAGE TO PROJECT
# Instead of an external image, we will generate a bright purple square 
# with some text on it to serve as our Augmented Reality overlay.
overlay = np.zeros((300, 300, 3), dtype="uint8")
overlay[:] = (255, 0, 128) # BGR color for Purple
cv2.putText(overlay, "HOMOGRAPHY!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

# Grab the height and width of our overlay image
(h, w) = overlay.shape[:2]

# Define the 4 corners of our virtual overlay image:
# [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
src_pts = np.array([
    [0, 0], 
    [w, 0], 
    [w, h], 
    [0, h]
], dtype="float32")

# 2. START THE CAMERA AND DETECTOR
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

print("[INFO] Starting Augmented Reality feed...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 3. MAIN VIDEO LOOP
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    (fH, fW) = frame.shape[:2]
    
    # Detect the ArUco marker
    (corners, ids, rejected) = detector.detectMarkers(frame)

    if len(corners) > 0:
        for markerCorner in corners:
            
            # The 4 corners of the ArUco marker in the live video
            dst_pts = markerCorner.reshape((4, 2))

            # =======================================================
            # THE MAGIC HOMOGRAPHY MATH
            # =======================================================
            # We ask OpenCV: "How do I stretch 'src_pts' so they perfectly 
            # match the slanted, squished shape of 'dst_pts'?"
            H, _ = cv2.findHomography(src_pts, dst_pts)

            # Now we use 'H' to physically stretch and warp our purple image!
            warped = cv2.warpPerspective(overlay, H, (fW, fH))

            # =======================================================
            # PASTE THE WARPED IMAGE ONTO THE CAMERA FEED
            # =======================================================
            # Create a blank black mask, and draw a white box exactly 
            # where the ArUco marker is.
            mask = np.zeros((fH, fW), dtype="uint8")
            cv2.fillConvexPoly(mask, dst_pts.astype("int32"), 255)
            
            # Invert the mask (Marker area becomes black, everything else white)
            mask_inv = cv2.bitwise_not(mask)
            
            # Black out the ArUco marker on our main camera frame
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            
            # Add the warped purple image into the blacked-out hole!
            frame = cv2.add(frame_bg, warped)

    cv2.imshow("Homography Augmented Reality", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cv2.destroyAllWindows()
vs.stop()