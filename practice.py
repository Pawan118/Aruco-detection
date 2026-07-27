import cv2
import numpy as np
import os
import glob
import imutils

ChessBoard_Size=(9,6)
Square_Size=2.0

if not os.path('calib_images'):
    os.markdirs('calib_images')
    
