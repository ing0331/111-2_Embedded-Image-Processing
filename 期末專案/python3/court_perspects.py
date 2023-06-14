
import cv2
import numpy as np

def perspective_transform_warp(video_path, output_path, src_points, dst_points):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

    # Initialize background subtraction
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        transformed_frame = apply_perspective_transform(frame, src_points, dst_points)

        cv2.imshow('Perspective Transform', transformed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def apply_perspective_transform(frame, src_points, dst_points):
    height, width = 1080, 555
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_frame = cv2.warpPerspective(frame, M, (width, height))
    return transformed_frame

# Define the source points (clockwise from top-left)
src_points = np.array([[444., 845.], [1477., 848.],  [633., 663.],[1286., 665.]], dtype="float32")
dst_points = np.array([[131.7, 633.], [398, 633.], [165., 500], [365., 500.]], dtype="float32")

# Input and output video paths
input_video_path = 'tennis_court.mp4'
output_video_path = 'bird_court.mp4'

perspective_transform_warp(input_video_path, output_video_path, src_points, dst_points)
