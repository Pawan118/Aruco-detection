import cv2

# 1. Load the exact same dictionary your drone script uses
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# 2. Generate the Markers (400x400 pixels)
marker_24 = cv2.aruco.generateImageMarker(arucoDict, 24, 400) # TARGET
marker_12 = cv2.aruco.generateImageMarker(arucoDict, 12, 400) # DECOY
marker_15 = cv2.aruco.generateImageMarker(arucoDict, 15, 400) # DECOY
marker_20 = cv2.aruco.generateImageMarker(arucoDict, 20, 400) # DECOY

# 3. Save them as PNG images
cv2.imwrite("marker_24_TARGET.png", marker_24)
cv2.imwrite("marker_12_DECOY.png", marker_12)
cv2.imwrite("marker_15_DECOY.png", marker_15)
cv2.imwrite("marker_20_DECOY.png", marker_20)

print("[INFO] Successfully saved all 4 markers!")
print("[INFO] Open them and print them on standard white paper (one per page).")
