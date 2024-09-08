import cv2
import numpy as np

weights_path = "C:/Users/gokul/Desktop/Computer_vision_mini_project/yolov4.weights"
config_path = "C:/Users/gokul/Desktop/Computer_vision_mini_project/yolov4.cfg"
names_path = "C:/Users/gokul/Desktop/Computer_vision_mini_project/coco.names"

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(50, 50),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Function to calculate speed in km/hr
def calculate_speed(prev_pts, curr_pts, fps, meters_per_pixel):
    displacement = np.linalg.norm(curr_pts - prev_pts, axis=1)
    speed = np.mean(displacement) * fps * meters_per_pixel * 3.6
    return speed

# Path to the video file
video_path = "C:/Users/gokul/Desktop/Computer_vision_mini_project/Cars Moving On Road Stock Footage - Free Download.mp4"

# Initialize video capture
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

ip_fps = cap.get(cv2.CAP_PROP_FPS)
if ip_fps == 0:
    ip_fps = 30

meters_per_pixel = 0.1

output_video_path = "output_video_1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(output_video_path, fourcc, ip_fps, (width, height))

# Debug: Check if VideoWriter is opened successfully
print(f"VideoWriter initialized: {video_writer.isOpened()}")

prev_pts = None
old_gray = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    yolo_outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in yolo_outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "car":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    new_pts = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            new_pts.append([x + w / 2, y + h / 2])
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if prev_pts is None:
        prev_pts = np.array(new_pts, dtype=np.float32).reshape(-1, 1, 2)
        old_gray = frame_gray.copy()
        continue

    curr_pts, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, prev_pts, None, **lk_params)
    if curr_pts is None:
        prev_pts = np.array(new_pts, dtype=np.float32).reshape(-1, 1, 2)
        old_gray = frame_gray.copy()
        continue

    good_new = curr_pts[st == 1]
    good_old = prev_pts[st == 1]

    average_speed = calculate_speed(good_old, good_new, ip_fps, meters_per_pixel)
    speed_text = f"Average speed: {average_speed:.2f} km/hr"
    cv2.putText(frame, speed_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    # Debug: Check the type of 'video_writer' before writing
    # print(f"Type of 'video_writer': {type(video_writer)}")

    video_writer.write(frame)  # Writing the frame to the output video
    # cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    old_gray = frame_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("Video processing complete.")
