import numpy as np
import cv2
import os

class Trajectory:

    def __init__(self, point):
        self.__points = []  # points 의 각 요소는 [2] shape의 numpy 배열이어야 함
        self.__length = 0
        self.__missing_count = 0

        self.__points.append(point)



    # Trajectory 포인트를 리턴하는 함수
    def getPoints(self):
        return self.__points


    # Trajectory 포인트를 추가하는 함수
    def addPoint(self, point):
        self.__points.append(point)
        self.__missing_count = 0

        last_points = self.__points[-2:]
        point_diff = last_points[1] - last_points[0]
        point_diff_mag = np.sqrt(point_diff.dot(point_diff))
        self.__length += point_diff_mag


    def getLength(self):
        return self.__length



    def checkNextPoint(self, point):

        points_length = len(self.__points)

        if points_length >= 3:
            # points 배열에 3개 이상의 포인트가 있는 경우
            # 다음 예측 포인트와 거리 확인
            #nextPoint = Trajectory.predictNextPoint(self.__points)
            #point_diff = point - nextPoint
            #point_diff_mag = np.sqrt(point_diff.dot(point_diff))
            #return (point_diff_mag < 8.0)
            return Trajectory.checkTriplet(self.__points[-2:] + [point])

        elif points_length == 0:
            # points 배열이 비어있는 경우
            # 무조건 True 리턴
            return True

        elif points_length == 1:
            # points 배열에 1개의 포인트만 있는 경우
            # 거리만 확인
            point_diff = point - self.__points[0]
            point_diff_mag = np.sqrt(point_diff.dot(point_diff))
            return (point_diff_mag > 2.0) and (point_diff_mag < 80.0)

        elif points_length == 2:
            # points 배열에 2개의 포인트가 있는 경우
            # Triplet 여부 확인
            return Trajectory.checkTriplet(self.__points + [point])



    # Missing 카운트 올리는 함수 -> 추적 계속 여부 리턴
    def upcountMissing(self):

        if len(self.__points) < 3:
            return False

        self.__missing_count += 1

        # missing count 초과 여부 확인
        if self.__missing_count > 1:
            # 추적 종료
            return False

        else:
            # 다음 추정 포인트 추가
            nextPoint = self.predictNextPoint(self.__points)
            self.__points.append(nextPoint)
            return True




    # 다음 포인트를 예측하여 리턴하는 함수
    @classmethod
    def predictNextPoint(self, points):

        if len(points) >= 3 :

            # 뒤에서 3개 포인트 추출
            last3Points = points[-3:]

            # 속도와 가속도 구함
            velocity = [last3Points[1] - last3Points[0], last3Points[2] - last3Points[1]]
            acceleration = velocity[1] - velocity[0]

            # 다음 위치 추정
            nextVelocity = velocity[1] + acceleration
            nextPoint = last3Points[2] + nextVelocity

            return nextPoint



    # 초기 유효 3포인트 만족 여부를 확인하는 함
    @classmethod
    def checkTriplet(self, points):

        if len(points) != 3:
            return False

        # 속도와 가속도 구함
        velocity = [points[1] - points[0], points[2] - points[1]]
        acceleration = velocity[1] - velocity[0]

        #print("acceleration :", acceleration)

        # 속도 크기가 비슷해야 함
        velocity_mag = [np.sqrt(velocity[0].dot(velocity[0])), np.sqrt(velocity[1].dot(velocity[1]))]
        if velocity_mag[0] > velocity_mag[1]:
            if velocity_mag[1] / velocity_mag[0] < 0.6:
                #print("velocity_mag[1] / velocity_mag[0] :", velocity_mag[1] / velocity_mag[0])
                return False
        else:
            if velocity_mag[0] / velocity_mag[1] < 0.6:
                #print("velocity_mag[0] / velocity_mag[1] :", velocity_mag[0] / velocity_mag[1])
                return False

        # 속도가 너무 작거나 크지 않아야 함
        if velocity_mag[0] < 2.0 or velocity_mag[0] > 80.0:
            #print("velocity_mag[0] :", velocity_mag[0])
            return False
        if velocity_mag[1] < 2.0 or velocity_mag[1] > 80.0:
            #print("velocity_mag[1] :", velocity_mag[1])
            return False

        # 속도 방향 변화가 작아야 함
        velocity_dot = velocity[1].dot(velocity[0])
        acceleration_angle = np.arccos(velocity_dot / (velocity_mag[0] * velocity_mag[1]))
        #print("acceleration_angle :",  np.rad2deg(acceleration_angle))
        if acceleration_angle > np.deg2rad(45.0):
            return False

        # 가속도가 작아야 함
        acceleration_mag = np.sqrt(acceleration.dot(acceleration))
        if acceleration_mag > 20.0:
            #print("acceleration_mag :", acceleration_mag)
            return False

        if acceleration[0] < -2.0:
            return False

        return True



class TrajectoryManager:

    def __init__(self):
        self.__trajectorys = []


    def getTrajectorys(self):
        return self.__trajectorys


    def setPointsFrame(self, points):

        max_trajectory = (0, 0)
        trajectorys_updated = [False] * len(self.__trajectorys)

        for index, point in enumerate(points):

            isAddedTrajectory = False

            # 기존 Trajectory에 추가되는 포인트인지 확인
            for index, updated in enumerate(trajectorys_updated):

                if updated == False:
                    if self.__trajectorys[index].checkNextPoint(point):
                        self.__trajectorys[index].addPoint(point)
                        trajectorys_updated[index] = True
                        isAddedTrajectory = True

                        trajectory_length = self.__trajectorys[index].getLength()
                        if trajectory_length > max_trajectory[0]:
                            max_trajectory = (trajectory_length, index)

                        break


            # Trajectory에 추가되지 않은 포인트는 신규 Trajectory로 생성
            if isAddedTrajectory == False:
                trajectory_new = Trajectory(point)
                self.__trajectorys.append(trajectory_new)


        # 높은 가능성의 Trajectory가 찾아지면 해당 Trajectory만 남김
        if max_trajectory[0] > 30.0:
            self.__trajectorys = [self.__trajectorys[max_trajectory[1]]]

        else:

            # 업데이트 되지 않은 Trajectory의 Missing Count 증가
            for index, updated in reversed(list(enumerate(trajectorys_updated))):

                if updated == False:
                    if self.__trajectorys[index].upcountMissing() == False:
                        self.__trajectorys.remove(self.__trajectorys[index])




print(os.getcwd())





bRecord = False
if bRecord == True:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('{}.avi'.format(video_path.split('/')[-1].split('.')[0]),fourcc, 20.0, (640,360))
    out = cv2.VideoWriter("RenderOutput.mp4", cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 20.0, (640,360))

cap = cv2.VideoCapture("D:\\Project_lab\\img_python\\tennis_video\\tennis_court.mp4")
bgSubtractor = cv2.createBackgroundSubtractorKNN(history = 10, dist2Threshold = 200.0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

kernel_size = 11
kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

frame_count = 0
trajectory_image = np.zeros([360, 640, 3], np.uint8)
point_image = np.zeros([360, 640, 3], np.uint8)


manager = TrajectoryManager()


while cap.isOpened() :

    ret, frame = cap.read()
    sh = np.shape(frame)
    resize_scale = 640. / float(sh[1])

    frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)

    print(frame.shape)


    # 가우시안 블러 적용
    blur = cv2.GaussianBlur(frame, (7, 7), 0)


    # Background 마스크 생성
    fgmask = bgSubtractor.apply(blur)
    blank_image = np.zeros(fgmask.shape, np.uint8)


    # Background 마스크에 모폴로지 적용
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel_dilation)



    point_image = cv2.addWeighted(point_image, 0.9, np.zeros(frame.shape, np.uint8), 0.1, 0)


    #print("frame_count :", frame_count)
    frame_count += 1

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

    #print(len(centroids))



    points = []
    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue


        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])


        area_ratio = area / (width * height)
        aspect_ratio = width / height
        #print(x, y, area, width * height, area_ratio)



        #if area > 2 and area < 2000:
        if area_ratio > 0.6 and aspect_ratio > 0.333 and aspect_ratio < 3.0 and area < 500 and fgmask[centerY, centerX] == 255:

            #cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)
            # cv2.rectangle(frame, (x-1, y-1), (x-1 + width+2, y-1 + height+2), (0, 0, 255))
            cv2.rectangle(frame, (x - 1, y - 1), (x - 1 + width + 2, y - 1 + height + 2), (0, 255, 0))

            point_image[centerY, centerX] = (255, 255, 255)
            points.append(np.array([centerY, centerX]))


            # 해당 포인트의 컬러 값 얻기
            for pixel_y in range(y, y + height):
                for pixel_x in range(x, x + width):

                    if fgmask[pixel_y, pixel_x] >= 0:
                        #frame[pixel_y, pixel_x] = [0, 255, 0]
                        blank_image[pixel_y, pixel_x] = 255

        #else :

         #   cv2.rectangle(frame, (x - 1, y - 1), (x - 1 + width + 2, y - 1 + height + 2), (0, 0, 255))


    manager.setPointsFrame(points)



    for trajectory in manager.getTrajectorys():

        points = trajectory.getPoints()

        #print(points)

        if len(points) < 3:
            continue


        for index, point in enumerate(points):
            # if point[0] < 360 and point[1] < 640:
                # trajectory_image[point[0], point[1]] = (0, 255, 0)
                # cv2.circle(frame, (point[1], point[0]), 1, (255, 255, 0), 2)

            if index >= 1:
                cv2.line(frame, (points[index-1][1], points[index-1][0]), (point[1], point[0]), (255, 255, 0), 1)



    #cv2.imshow('processed', fgmask)
    #cv2.imshow('point', point_image)
    cv2.imshow('raw', frame)

    # record
    if bRecord == True:
        out.write(frame)

    # terminate
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break


if bRecord == True:
    out.release()

cap.release()
cv2.destroyAllWindows()