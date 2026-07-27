import numpy as np

# Python can read the binary files and print them as text for you!
matrix = np.load('camera_matrix.npy')
distortion = np.load('dist_coeffs.npy')

print("--- Camera Matrix (Focal Length & Center) ---")
print(matrix)
print("\n--- Distortion Coefficients (Lens Curve) ---")
print(distortion)