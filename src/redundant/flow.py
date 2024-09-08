import cv2

import numpy as np

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(60, 60),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Function to calculate speed in km/hr
def calculate_speed(prev_pts, curr_pts, fps, meters_per_pixel):
    # Calculate the displacement (difference in position)
    displacement = np.linalg.norm(curr_pts - prev_pts, axis=1)
    # Calculate speed (distance per frame) and convert to km/hr
    speed = np.mean(displacement) * fps * meters_per_pixel * 3.6
    return speed

# Path to the video file
video_path = "C:/Users/gokul/Desktop/New folder (2)/car passing by.mp4"

# Initialize video capture
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Take the first frame and find features in it
ret, old_frame = cap.read()
if not ret:
    print("Error: Failed to capture the first frame.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
if fps == 0:
    fps = 30  # Default to 30 FPS if unable to get FPS from video capture

# Meter per pixel scale factor (adjust based on video resolution and real-world scale)
# You may need to manually measure the scale of objects in the video
meters_per_pixel = 0.2  # For example, assuming 1 pixel represents 0.1 meters (10 cm)

# Initialize video writer for output video
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    curr_pts, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, prev_pts, None, **lk_params)
    
    if curr_pts is None:
        # If optical flow calculation fails, skip this frame
        prev_pts = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        old_gray = frame_gray.copy()
        continue

    # Select good points
    good_new = curr_pts[st == 1]
    good_old = prev_pts[st == 1]

    # Calculate speed
    average_speed = calculate_speed(good_old, good_new, fps, meters_per_pixel)
    
    # Overlay speed on the frame with increased font size
    speed_text = f"Average speed: {average_speed:.2f} km/hr"
    cv2.putText(frame, speed_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    # Write frame to output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete.")
