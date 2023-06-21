import numpy as np
import cv2

# Open the video file
cap = cv2.VideoCapture("tennis_court.mp4")
player_count = 0
# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Get the original video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the output video
output_filename = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Set the desired width and height for the output window
output_width = 640
output_height = 480

# Create a resizable output window
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# Resize the output window
cv2.resizeWindow("Output", output_width, output_height)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(8, 8),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
)

# Initialize previous frame and keypoints
prev_frame = None
prev_keypoints = []

# Initialize bounding boxes for objects
boxes = []

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        break

    # Define the list of boundaries
    boundaries = [([180, 180, 100], [255, 255, 255])]

    # Loop over the boundaries
    for (lower, upper) in boundaries:
        # Create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # Find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # Apply corner detection
    corners = cv2.cornerHarris(gray, 9, 3, 0.01)
    corners = cv2.normalize(corners, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    _, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Current keypoints
    keypoints = []

    for contour in contours:
        center, _, _ = cv2.minAreaRect(contour)
        center = (int(center[0]), int(center[1]))
        keypoints.append(center)
        cv2.circle(frame, center, 3, (0, 0, 255), -1)

    # Optical flow
    if prev_frame is not None:
        prev_keypoints = np.float32(prev_keypoints).reshape(-1, 1, 2)
        next_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_keypoints, None, **lk_params)
        next_keypoints = next_keypoints.reshape(-1, 2)

        # Reset bounding boxes for each frame
        boxes = []

        for i, (prev_pt, next_pt) in enumerate(zip(prev_keypoints, next_keypoints)):
            x1, y1 = prev_pt.ravel()
            x2, y2 = next_pt.ravel()

            # Compute the distance between the previous and current positions
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Draw the tail if the movement is significant
            if distance > 1:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Update bounding boxes
            if distance > 2:
                # Find the minimum and maximum coordinates for the bounding box
                min_x = int(min(x1, x2))
                min_y = int(min(y1, y2))
                max_x = int(max(x1, x2))
                max_y = int(max(y1, y2))

                # Check if the current bounding box overlaps with existing boxes
                overlap = False
                for box in boxes:
                    if min_x <= box[2] and max_x >= box[0] and min_y <= box[3] and max_y >= box[1]:
                        overlap = True
                        break

                # Add the new bounding box if no overlap is found
                if not overlap:
                    boxes.append([min_x, min_y, max_x, max_y])

    # Combine close bounding boxes
    combined_boxes = []
    while len(boxes) > 0:
        box = boxes.pop(0)
        min_x, min_y, max_x, max_y = box

        # Find close boxes
        close_boxes = [box]
        for other_box in boxes:
            other_min_x, other_min_y, other_max_x, other_max_y = other_box
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            other_center_x = (other_min_x + other_max_x) / 2
            other_center_y = (other_min_y + other_max_y) / 2

            # Compute the distance between the centers of the boxes
            distance = np.sqrt((other_center_x - center_x) ** 2 + (other_center_y - center_y) ** 2)

            if distance < 200:
                close_boxes.append(other_box)

        # Compute the combined bounding box
        combined_box = [
            min(min_x for min_x, _, _, _ in close_boxes),
            min(min_y for _, min_y, _, _ in close_boxes),
            max(max_x for _, _, max_x, _ in close_boxes),
            max(max_y for _, _, _, max_y in close_boxes),
        ]
        # Calculate the area of the combined bounding box
        combined_area = (combined_box[2] - combined_box[0]) * (combined_box[3] - combined_box[1])

        # Initialize an empty array to store the numbers for combined boxes
        combined_numbers = []

        # Loop through the combined bounding boxes
        for box in combined_boxes:
            # Calculate the area of the box
            area = (box[2] - box[0]) * (box[3] - box[1])

            # Check if the area is greater than 500 pixels
            if area > 250:
                # Generate a unique number for the box if not already assigned
                if box not in combined_numbers:
                    number = len(combined_numbers) + 1
                    combined_numbers.append(box)
                else:
                    # Find the existing number assigned to the box
                    number = combined_numbers.index(box) + 1

                # Draw the combined bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                # Display the number on the combined bounding box
                cv2.putText(frame, f"Player {number}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

        combined_boxes.append(combined_box)

        # Remove the close boxes from the original list
        boxes = [box for box in boxes if box not in close_boxes]

    # Draw bounding boxes
    for box in combined_boxes:
        min_x, min_y, max_x, max_y = box
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    # Print the positions and number of keypoints
    for i, keypoint in enumerate(keypoints):
        print(f"Keypoint {i + 1}: Position: {keypoint}")

    print(f"Total number of keypoints: {len(keypoints)}")

    # Add text to the output frame
    text = f"Total keypoints: {len(keypoints)}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize the frame to fit the output window
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Display the resized frame in the output window
    cv2.imshow("ply_boxes", resized_frame)

    # Update previous frame and keypoints
    prev_frame = gray.copy()
    prev_keypoints = keypoints

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Write the resized frame to the output video file
out.write(resized_frame)

# Release the video file and close windows
cap.release()
cv2.destroyAllWindows()
