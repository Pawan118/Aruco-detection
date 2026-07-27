import cv2
import numpy as np
import os
import glob

# --- SETTINGS ---
# We are using a 10x7 checkerboard, which has 9x6 inner corners
CHESSBOARD_SIZE = (9, 6) 
SQUARE_SIZE = 3.0 # Physical size of the square (doesn't heavily matter for this phase)

# Create a folder to save our pictures
if not os.path.exists('calib_images'):
    os.makedirs('calib_images')

# Prepare the perfect 3D grid in the computer's memory
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

print("[INFO] Starting camera...")
print("[INFO] Press 's' to SAVE an image (Try to get 15-20 images from different angles).")
print("[INFO] Press 'c' to COMPUTE the calibration.")
print("[INFO] Press 'q' to QUIT.")

cap = cv2.VideoCapture(0)
img_count = 0

# --- PHASE 1: TAKE PICTURES ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Look for the checkerboard
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    display_frame = frame.copy()
    if ret_corners:
        # Draw rainbow lines over the corners if found
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)
        cv2.putText(display_frame, "BOARD DETECTED - Press 's' to save!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "Looking for board...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.putText(display_frame, f"Images Saved: {img_count}/15+", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow('Camera Calibration', display_frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Save the frame
    if key == ord('s') and ret_corners:
        img_name = f"calib_images/calib_{img_count}.png"
        cv2.imwrite(img_name, frame)
        print(f"[SAVED] {img_name}")
        img_count += 1
        
    # Start computing
    elif key == ord('c'):
        if img_count < 10:
            print("[WARNING] You should take at least 10 images before calibrating!")
        else:
            break
            
    # Quit safely
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# --- PHASE 2: DO THE MATH ---
cap.release()
cv2.destroyAllWindows()

print("\n[INFO] Calculating camera distortion... this might take a few seconds.")

# Load all the images we just saved
images = glob.glob('calib_images/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    if ret_corners:
        # Refine corner locations for extreme accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        objpoints.append(objp)
        imgpoints.append(corners2)

print(f"[INFO] Processing {len(objpoints)} valid images...")

# Calculate the actual distortion!
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("RMS Reprojection Error:", ret)
# Save the mathematical results to files
np.save('camera_matrix.npy', cameraMatrix)
np.save('dist_coeffs.npy', distCoeffs)


print("\n=== CALIBRATION SUCCESSFUL ===")
print("Saved 'camera_matrix.npy' and 'dist_coeffs.npy' to your folder.")
