
import numpy as np
import cv2

# Global variables
drawing = False  # True if mouse is pressed for ROI selection
ix, iy = -1, -1  # Initial coordinates of the mouse
reset = False    # True if reset button is pressed
keypoints = []   # List to store keypoints
def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, reset

    if reset:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Image", image)

        # Capture keypoints within the ROI
        roi = image[iy:y, ix:x]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        corners = cv2.cornerHarris(gray, 9, 3, 0.01)
        corners = cv2.normalize(corners, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        _, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            center, _, _ = cv2.minAreaRect(contour)
            center = (int(center[0]) + ix, int(center[1]) + iy)
            keypoints.append(center)
            cv2.circle(image, center, 3, (0, 0, 255), -1)
            cv2.putText(image, str(i+1), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Print the positions and number of keypoints within the ROI
        print(f"Keypoints in ROI:")
        for i, keypoint in enumerate(keypoints):
            print(f"Keypoint {i+1}: Position: {keypoint}")
        print(f"Total number of keypoints in ROI: {len(keypoints)}")

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_roi)

def reset_image():
    global image, reset, keypoints
    image = cv2.imread("tennis_court.jpg")
    reset = True
    keypoints = []
    cv2.imshow("Image", image)
    reset = False

reset_image()

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):  # Reset button
        reset_image()

    elif key == ord("q"):  # Quit button
        break

    if reset:
        continue

    # Display the image with keypoints and ROI
    cv2.imshow("Image", image)

cv2.destroyAllWindows()
