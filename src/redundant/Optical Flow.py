import cv2
import numpy as np

def calculate_optical_flow(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        return
    
    # Convert the frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Define the parameters for the optical flow
    params = {
        'pyr_scale': 0.5,  # Pyramid scale (0.5 means a classical pyramid, where each level is half the resolution of the previous one)
        'levels': 3,       # Number of pyramid levels
        'winsize': 15,     # Averaging window size
        'iterations': 3,   # Number of iterations at each pyramid level
        'poly_n': 5,       # Size of the pixel neighborhood used to find polynomial expansion in each pixel
        'poly_sigma': 1.2, # Standard deviation of the Gaussian used to smooth derivatives
        'flags': 0         # Operation flags
    }

    # Loop over the frames of the video
    while True:
        # Capture the next frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute the dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **params)
        
        # Compute the magnitude and angle of the flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate the average magnitude of the flow vectors
        avg_magnitude = np.mean(magnitude)
        
        # Print or log the average magnitude as an estimate of speed
        print(f"Average speed: {avg_magnitude:.2f} units/frame")
        
        # Update the previous frame and previous gray frame
        prev_gray = gray
    
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

# Path to your video file
video_path = 'path_to_your_video.mp4'
calculate_optical_flow(video_path)
