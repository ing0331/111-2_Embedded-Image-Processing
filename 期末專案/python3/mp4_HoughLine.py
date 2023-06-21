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

    # Show the result with resized output video
    resized_output = cv2.resize(line_image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('line_intersections', resized_output)
    cv2.waitKey(0)

    # Mark intersections on the resized output video
    for inter in intersections:
        a, b = inter
        a = int(a)
        b = int(b)
        if a >= 0 and a < resized_output.shape[1] and b >= 0 and b < resized_output.shape[0]:
            for i in range(6):
                for j in range(6):
                    if b + i < resized_output.shape[0] and a + j < resized_output.shape[1]:
                        resized_output[b + i, a + j] = [0, 0, 255]

    # Show the resized output video with marked intersections
    cv2.imshow('line_intersections_marked', resized_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
