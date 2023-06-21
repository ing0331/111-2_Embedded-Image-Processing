import numpy as np
import cv2

# Open the video file
cap = cv2.VideoCapture("tennis_court.mp4")

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Get the dimensions of the input video
input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_center = (input_width // 2, input_height // 2)
print(input_height, input_width)
# Set the desired width and height for the output window
output_width = 1280
output_height = 720

# Create a resizable output window
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# Resize the output window
cv2.resizeWindow("Output", output_width, output_height)

# Variables to track the court center keypoints, nearby keypoints, and bottom keypoints
court_center_keypoints = []
nearby_keypoints = []
bottom_keypoints = []
src_points = np.array([nearby_keypoints, bottom_keypoints], dtype="float32")

radius = 25

# Flag to indicate whether to lock the nearby keypoints and bottom keypoints
lock_nearby_keypoints = False
lock_bottom_keypoints = False
lock_count = 0

# Loop through the video frames
frame_count = 0
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

    # Start your code
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    corners = cv2.cornerHarris(gray, 9, 3, 0.01)
    corners = cv2.normalize(corners, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    _, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    keypoints = []
    for contour in contours:
        center, _, _ = cv2.minAreaRect(contour)
        center = (int(center[0]), int(center[1]))

        keypoints.append(center)
        cv2.circle(frame, center, 3, (0, 0, 255), -1)

    # Print the positions and number of keypoints
    for i, keypoint in enumerate(keypoints):
        print(f"Keypoint {i+1}: Position: {keypoint}")

    print(f"Total number of keypoints: {len(keypoints)}")

    # Track the three keypoints closest to the most middle position
    court_center_keypoints = sorted(keypoints, key=lambda kp: abs(kp[0] - input_center[0]))[:3]

    # Lock the nearby keypoints if they don't move for the initial three frames
    if not lock_nearby_keypoints and frame_count < 3:
        if len(court_center_keypoints) >= 3:
            third_keypoint = court_center_keypoints[1]
            down_mid_Keypoint = court_center_keypoints[0]
            temp_nearby_keypoints = [kp for kp in keypoints if abs(kp[1] - third_keypoint[1]) <= 5 and kp != third_keypoint][:2]

            if nearby_keypoints:
                differences = [abs(temp_nearby_keypoints[i][0] - nearby_keypoints[i][0]) +
                               abs(temp_nearby_keypoints[i][1] - nearby_keypoints[i][1]) for i in range(2)]

                if max(differences) <= 5:
                    lock_count += 1
                else:
                    lock_count = 0

                if lock_count >= 3:
                    lock_nearby_keypoints = True

            nearby_keypoints = temp_nearby_keypoints

    # Lock the bottom keypoints if they don't move for the initial three frames
    if not lock_bottom_keypoints and frame_count < 3:
        temp_bottom_keypoints = [kp for kp in keypoints if abs(kp[1] - down_mid_Keypoint[1]) <= 2 and kp != down_mid_Keypoint][:2]

        if bottom_keypoints:
            differences = [abs(temp_bottom_keypoints[i][0] - bottom_keypoints[i][0]) +
                           abs(temp_bottom_keypoints[i][1] - bottom_keypoints[i][1]) for i in range(2)]

            if max(differences) <= 5:
                lock_count += 1
            else:
                lock_count = 0

            if lock_count >= 3:
                lock_bottom_keypoints = True

        bottom_keypoints = temp_bottom_keypoints

    # Print the nearby keypoints
    for i, nearby_keypoint in enumerate(nearby_keypoints):
        print(f"Nearby Keypoint {i+1}: Position: {nearby_keypoint}")

    # Print the bottom keypoints
    for i, bottom_keypoint in enumerate(bottom_keypoints):
        print(f"Bottom Keypoint {i+1}: Position: {bottom_keypoint}")

    # Draw orange circles around the court center keypoints
    for court_center_keypoint in court_center_keypoints:
        cv2.circle(frame, court_center_keypoint, radius, (0, 165, 255), -1)

    # Draw green circles around the nearby keypoints
    for nearby_keypoint in nearby_keypoints:
        cv2.circle(frame, nearby_keypoint, radius, (0, 255, 0), -1)

    # Draw purple circles around the bottom keypoints
    for bottom_keypoint in bottom_keypoints:
        cv2.circle(frame, bottom_keypoint, radius, (255, 0, 255), -1)

    # Resize the frame to fit the output window
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Display the resized frame in the output window
    cv2.imshow("Output", resized_frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release the video file and close windows
cap.release()
cv2.destroyAllWindows()
