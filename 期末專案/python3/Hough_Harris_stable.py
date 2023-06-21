import numpy as np
import cv2

def get_line_intersection(line1, line2):
    """Find the intersection point of two lines defined by endpoints."""
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines are parallel or coincident

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

# load the video
cap = cv2.VideoCapture("tennis_court.mp4")

# create VideoWriter object to save the output frames into a video file
output_file = "output_video.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2  # Resize width
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2  # Resize height
output = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (frame_width, frame_height))

# define the list of boundaries
boundaries = [
    ([180, 180, 100], [255, 255, 255])
]

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (frame_width, frame_height))

    # Apply color mask
    mask = np.zeros_like(resized_frame)
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(resized_frame, lower, upper)
    output_frame = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)

    gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 10
    high_threshold = 200
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    dilated = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8))

    rho = 1
    theta = np.pi / 180
    threshold = 10
    min_line_length = 40
    max_line_gap = 5
    line_image = np.copy(output_frame) * 0

    lines = cv2.HoughLinesP(dilated, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    intersections = []
    for idx, point1 in enumerate(points):
        for point2 in points[idx + 1:]:
            intersect = get_line_intersection(point1, point2)
            if intersect is not None:
                intersections.append(intersect)

    # Perform Harris corner detection
    corners = cv2.cornerHarris(blur_gray, 9, 3, 0.01)
    corners = cv2.normalize(corners, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    _, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dilated_corners = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated_corners, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    keypoints = []
    for contour in contours:
        center, _, _ = cv2.minAreaRect(contour)
        center = (int(center[0]), int(center[1]))
        keypoints.append(center)
        cv2.circle(resized_frame, center, 3, (0, 0, 255), -1)

    # Find keypoints overlapping with Hough lines
    overlapping_keypoints = []
    for keypoint in keypoints:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if cv2.pointPolygonTest(contour, keypoint, False) >= 0:
                    overlapping_keypoints.append(keypoint)
                    break

    # Print the positions and number of overlapping keypoints
    for i, keypoint in enumerate(overlapping_keypoints):
        print(f"Keypoint {i+1}: Position: {keypoint}")

    print(f"Total number of overlapping keypoints: {len(overlapping_keypoints)}")

    # Add text to the output frame
    text = f"Total overlapping keypoints: {len(overlapping_keypoints)}"
    cv2.putText(resized_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize the frame to fit the output window
    resized_output_frame = cv2.resize(resized_frame, (640, 480))

    # Display the resized frame in the output window
    cv2.imshow("Output", resized_output_frame)

    # Write the resized frame to the output video file
    output.write(resized_output_frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close windows
cap.release()
output.release()
cv2.destroyAllWindows()
